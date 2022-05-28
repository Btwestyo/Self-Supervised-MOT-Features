import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision.models as models

import cv2
import os
import sys

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.models.model_convresnet import CustomResNetConv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PuzzleNet(nn.Module):
    def __init__(self, checkpoint_filename, hidden_dim = 512):
        super(PuzzleNet, self).__init__()

        self.imported = CustomResNetConv(hidden_dim)

        # load saved checkpoint
        m_dict = torch.load(checkpoint_filename)

        # load model state
        self.imported.load_state_dict(m_dict['model_state_dict'])

        self.model = torch.nn.Sequential(*(list(self.imported.resnet.children())), *(list(self.imported.custom_layers.children())[:-3]))

        self.model.requires_grad_ = False

        self.model.eval()

        # self.model.to(DEVICE)

    def forward(self,x):

        return self.model(x)

if __name__ == '__main__':

    hd   = 128
    ckpt = '/home/bcollico/github/cs231n_psets/final_project/scripts/training/checkpoints/mars_puzzle_'+str(hd)+'_ckpt.pt'

    model = PuzzleNet(ckpt, hd)

    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    