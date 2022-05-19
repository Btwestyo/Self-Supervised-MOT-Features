import torch
import numpy as np
import os
import sys
import cv2

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from matplotlib import pyplot as plt

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.models.model_example import ConvClassifier
from scripts.solver_mrunal import Solver
from scripts.plotting.plot_results import plot_loss_and_acc
from scripts.models.model_mrunal import CustomResNet

from scripts.dataloaders.dataloader_mrunal import RotationDataSet

if __name__ == "__main__":
    # instantiate model, optimizer, and loss function for solver
    model = CustomResNet()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=2e-4)
    loss_fcn = torch.nn.CrossEntropyLoss()

    solver = Solver(model, optimizer, loss_fcn)

    # get training and test datasets from torch
    training_data = RotationDataSet("/home/peanut/Documents/cs231n_final_project/data/detection_bounding_boxes/train")
    test_data = RotationDataSet("/home/peanut/Documents/cs231n_final_project/data/detection_bounding_boxes/test")

    # create dataloaders for datasets
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Viz a sample
    # for sample in train_dataloader:
    #   for i in range(64):
    #     import pdb; pdb.set_trace()
    #     plt.imshow(sample[0][i, :, :, :], interpolation='nearest')
    #     plt.show()
    #     import pdb; pdb.set_trace()

    # train the model
    solver.train(epochs=10,
                train_data_loader=train_dataloader,
                val_data_loader=test_dataloader,
                save_path=None,#os.path.join(SCRIPT_DIR,'checkpoints'),
                save_every=100,
                print_every=1,
                verbose=True)

    # plot training/validation accuracy and loss
    plot_loss_and_acc(solver.train_loss_history, solver.train_acc_history)
    plot_loss_and_acc(solver.val_loss_history, solver.val_acc_history)

    # save final solver object
    solver.save_solver(os.path.join(SCRIPT_DIR,'checkpoints','example_trained_model.pt'))