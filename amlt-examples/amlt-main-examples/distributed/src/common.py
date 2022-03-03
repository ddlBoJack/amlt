#!/usr/bin/env python

import os
import torch
import torch.nn as nn


def get_args():
  envvars = ['WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'NODE_RANK', 'NODE_COUNT', 'HOSTNAME', 'MASTER_ADDR', 'MASTER_PORT',
             'NCCL_SOCKET_IFNAME', 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE', 'OMPI_COMM_WORLD_LOCAL_RANK',
             'AZ_BATCHAI_MPI_MASTER_NODE']
  args = dict(gpus_per_node=torch.cuda.device_count())
  missing = []
  for var in envvars:
    if var in os.environ:
      args[var] = os.environ.get(var)
      try:
        args[var] = int(args[var])
      except ValueError:
        pass
    else:
      missing.append(var)
  print(f"II Args: {args}")
  if missing:
    print(f"II Environment variables not set: {', '.join(missing)}.")
  return args


class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = nn.Linear(10, 10)
    self.relu = nn.ReLU()
    self.net2 = nn.Linear(10, 5)

  def forward(self, x):
    return self.net2(self.relu(self.net1(x)))


class ToyMpModel(nn.Module):
  def __init__(self, *devices):
    super(ToyMpModel, self).__init__()
    self.devices = devices
    self.net0 = torch.nn.Linear(10, 10).to(devices[0])
    self.net1 = torch.nn.Linear(10, 10).to(devices[1])
    self.net2 = torch.nn.Linear(10, 10).to(devices[2])
    self.net3 = torch.nn.Linear(10, 5).to(devices[3])
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.relu(self.net0(x.to(self.devices[0])))
    x = self.relu(self.net1(x.to(self.devices[1])))
    x = self.relu(self.net2(x.to(self.devices[2])))
    return self.net3(x.to(self.devices[3]))
