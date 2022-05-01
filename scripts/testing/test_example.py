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

from scripts.models.model_example import ConvClassifier
from scripts.solver import Solver
from scripts.plotting.plot_results import plot_loss_and_acc

if __name__ == "__main__":

    s_dict = torch.load(os.path.join(SCRIPT_DIR,'..','training','checkpoints','example_trained_model.pt'))
    solver = s_dict['solver']

    test_data = datasets.FashionMNIST(
                    root=os.path.join(SCRIPT_DIR,'..','..','data'),
                    train=False,
                    download=True,
                    transform=ToTensor()
                )

    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    test_loss, test_acc = solver.test(test_dataloader)

    plot_loss_and_acc(test_loss, test_acc)