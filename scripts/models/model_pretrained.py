import torch
import torch.nn as nn
import torchvision.models as models

import os
import sys

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PreTrainedResNet(nn.Module):
    def __init__(self, ckpt=None, hidden_dim=None):
        super(PreTrainedResNet, self).__init__()
        self.model =  models.resnet18(pretrained=True)

        # remove the resnet linear layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]), torch.nn.Flatten())

        self.model.requires_grad_ = False

        self.model.eval()

        self.model.to(DEVICE)

    def forward(self,x):

        return self.model(x)

if __name__ == '__main__':

    hd   = 1028
    ckpt = '/home/bcollico/github/cs231n_psets/final_project/scripts/training/checkpoints/mars_puzzle_'+str(1024)+'_ckpt.pt'

    model = PreTrainedResNetConv()

    print(model)

    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    