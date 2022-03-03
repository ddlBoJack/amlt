import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

SW = SummaryWriter(os.environ.get('AMLT_OUTPUT_DIR', '.'), flush_secs=30)


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Template Finetune Example')
parser.add_argument('model', type=str, help='path of the model to finetune')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# output
parser.add_argument('-o', '--output-dir', type=str, default=os.getenv('AMLT_OUTPUT_DIR', '/tmp'))

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = torch.load(args.model)
print("Model properly loaded")
with open(os.path.join(args.output_dir, 'stdout.txt'), 'w') as f:
  f.write("Model properly loaded")

for epoch in range(10):
    SW.add_scalar('accuracy/test', 10. * epoch, epoch)

torch.save(model, args.output_dir + "/finetuned_model.amlt")
