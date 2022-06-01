import time
from os.path import join
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torchvision.models as models

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

# from scripts.models.model_conv_ae import Conv_AE
# from scripts.models.model_ae import AE
from scripts.solver import Solver

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ConvAeMot(nn.Module):
    
    def __init__(self, ckpt):
        super(ConvAeMot, self).__init__()

        from scripts.models.model_conv_ae import Conv_AE

        self.imported = Conv_AE()

        # load saved checkpoint
        m_dict = torch.load(ckpt)

        # load model state
        self.imported.load_state_dict(m_dict['model_state_dict'])

        self.model = self.imported.encoder

        del self.imported

        self.model.requires_grad_ = False

        self.model.eval()

        self.model.to(DEVICE)

    def forward(self,x):
        return self.model(x).flatten()

class AeMot(nn.Module):
    
    def __init__(self, ckpt):
        super(AeMot, self).__init__()

        from scripts.models.model_ae import Conv_AE

        self.imported = Conv_AE()

        # load saved checkpoint
        m_dict = torch.load(ckpt)

        # load model state
        self.imported.load_state_dict(m_dict['model_state_dict'])

        self.imported.fc2 = list(self.imported.fc2.children())[0]

        self.model = torch.nn.Sequential(self.imported.encoder, nn.Flatten(), self.imported.fc2)

        del self.imported

        self.model.requires_grad_ = False

        self.model.eval()

        self.model.to(DEVICE)

    def forward(self,x):
        return self.model(x)

if __name__ == '__main__':

    # from scripts.models.model_conv_ae import Conv_AE
    from scripts.models.model_ae import Conv_AE

    ckpt = './scripts/training/checkpoints/ae_linear_full_coco_112.pt'

    solver = Solver(model=None, optimizer=None, loss_fcn=None)
    s_dict = torch.load(ckpt)
    solver = s_dict['solver']

    solver.save_checkpoint('./scripts/training/checkpoints/ae_linear_full_coco_112_ckpt.pt', -1)

    # model = ConvAeMot('./scripts/training/checkpoints/autoencoder_coco_224_ckpt.pt')
    model = AeMot('./scripts/training/checkpoints/ae_linear_full_coco_112_ckpt.pt')
    x = torch.rand(1,3,112,112)

    y = model(x)

    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")

    # print(model)

    print(y.size())