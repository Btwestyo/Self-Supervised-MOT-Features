import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import sys

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.dataloaders.dataloader_mars_puzzle import MarsDataSetPuzzle
from scripts.models.model_convresnet import CustomResNetConv
from scripts.solver import Solver
from scripts.plotting.plot_results import plot_loss_and_acc

if __name__ == "__main__":

    s_dict = torch.load(os.path.join(SCRIPT_DIR,'..','training','checkpoints','example_trained_model.pt'))
    solver = s_dict['solver']

    # get training and test datasets from torch
    test_data =  MarsDataSetPuzzle(
        os.path.join(SCRIPT_DIR,'..','..','data','mars','bbox_train'),
        overfit=True)

    test_dataloader = DataLoader(test_data, batch_size=50, shuffle=True)

    test_loss, test_acc = solver.test(test_dataloader, visualize=True)

    plot_loss_and_acc(test_loss, test_acc)