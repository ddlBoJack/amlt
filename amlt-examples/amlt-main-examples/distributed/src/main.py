#!/usr/bin/env python

import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from common import ToyModel, get_args


def demo_basic(gpu_rank, node_rank, gpus_per_node, world_size, master_uri):
  # This script is running on every gpu of every node
  global_rank = node_rank * gpus_per_node + gpu_rank
  print(f"II Running basic DDP example with gpu_rank={gpu_rank}, node_rank={node_rank}, global_rank={global_rank}"
        f" gpus_per_node={gpus_per_node}, world_size={world_size}")

  def generate_data(model, seed):
    torch.manual_seed(seed)
    outputs = model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(gpu_rank)
    return outputs, labels

  # ToyModels initialized from different random states
  torch.manual_seed(global_rank)
  model = ToyModel()
  torch.cuda.set_device(gpu_rank)
  model.cuda(gpu_rank)
  loss_fn = nn.MSELoss().cuda(gpu_rank)
  optimizer = optim.SGD(model.parameters(), lr=0.001)

  # Once synchronized, model is broadcasted from process where global_rank=0. Loss printed is the same.
  model = DDP(model, device_ids=[gpu_rank])
  outputs, labels = generate_data(model, seed=123)
  print(loss_fn(outputs, labels))

  # Training step from different 'minibatches'. Loss will be different.
  optimizer.zero_grad()
  outputs, labels = generate_data(model, seed=global_rank)
  loss_fn(outputs, labels).backward()
  optimizer.step()
  print(loss_fn(outputs, labels))

  # After the backprop got synchronized, model returns the same loss
  outputs, labels = generate_data(model, seed=321)
  print(loss_fn(outputs, labels))
  dist.barrier()


def demo_checkpoint(gpu_rank, node_rank, gpus_per_node, world_size, master_uri):
  """
  This script is running on every gpu of every node.
  We will be synchronizing the models manually, and so we will not wrap our model with DDP.
  """
  global_rank = node_rank * gpus_per_node + gpu_rank
  print(f"II Running checkpointing DDP example with gpu_rank={gpu_rank}, node_rank={node_rank}, global_rank={global_rank}.")

  torch.manual_seed(global_rank)
  model = ToyModel()
  torch.cuda.set_device(gpu_rank)
  model.cuda(gpu_rank)

  loss_fn = nn.MSELoss().cuda(gpu_rank)
  optimizer = optim.SGD(model.parameters(), lr=0.001)

  BROADCASTED_TENSOR = torch.tensor([global_rank + 1]).cuda(gpu_rank)  # Will be shared across all nodes and GPUs
  CHECKPOINT_PATH = tempfile.gettempdir() + "/code/model.checkpoint"  # Will be shared within the nodes across their GPUs

  if gpu_rank == 0:
    torch.save(model.state_dict(), CHECKPOINT_PATH)

  if global_rank == 0:
    BROADCASTED_TENSOR *= 100

  dist.broadcast(BROADCASTED_TENSOR, src=0)
  print(f" <Global_rank={global_rank}> {str(BROADCASTED_TENSOR)}")  # Will all be equal to torch.tensor([100], device='cuda:X')

  # Use a barrier() to make sure that processes > 0 loads the model after process 0 saves it.
  dist.barrier()
  model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location={'cuda:0': f'cuda:{gpu_rank}'}))
  optimizer.zero_grad()

  torch.manual_seed(123)
  outputs = model(torch.randn(20, 10).to(gpu_rank))
  labels = torch.randn(20, 5).to(gpu_rank)
  loss_fn(outputs, labels).backward()
  optimizer.step()

  # Every models on every GPUs were initialized differently, but their state dict was synchronized node-wise.
  # The losses will be equal among nodes but different between nodes.
  print(f" <Global_rank={global_rank}> {loss_fn(outputs, labels).item()}")

  # Use a barrier() to make sure that all processes have finished reading the checkpoint
  dist.barrier()

  if gpu_rank == 0:
    os.remove(CHECKPOINT_PATH)


if __name__ == "__main__":
  # every nodes are executing their copy of this script
  args = get_args()

  master_uri = "tcp://%s:%s" % (args.get('MASTER_ADDR'), args.get('MASTER_PORT'))
  os.environ['NCCL_DEBUG'] = 'WARN'

  node_count = args.get('WORLD_SIZE')  # world size envvar only counts the number of nodes
  node_rank = args.get('NODE_RANK')

  gpus_per_node = torch.cuda.device_count()
  world_size = gpus_per_node * node_count
  gpu_rank = args.get('LOCAL_RANK')
  global_rank = node_rank * gpus_per_node + gpu_rank

  dist.init_process_group(backend='nccl', init_method=master_uri, world_size=world_size, rank=global_rank)
  demo_basic(gpu_rank=gpu_rank, node_rank=args.get('NODE_RANK'), gpus_per_node=gpus_per_node,
             world_size=world_size, master_uri=master_uri)
  demo_checkpoint(gpu_rank=gpu_rank, node_rank=args.get('NODE_RANK'), gpus_per_node=gpus_per_node,
                  world_size=world_size, master_uri=master_uri)
  dist.destroy_process_group()
