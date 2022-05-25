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

def train_gridsearch():
    results = []

    best_model = None
    best_val_acc = None
    best_model_epoch_idx = None

    lrs = [0.00104, 0.00154] #np.random.uniform(1e-4, 1e-2, 10)
    beta1s =  [0.9034] #np.random.uniform(0.88, 0.92, 5)
    beta2s = [0.997] #np.random.uniform(0.97, 0.999, 5)
    batch_sizes = [4, 8]# (4 * np.random.randint(1, 12, 6)).astype(np.int64)
    weight_decay = 0.00226 #np.random.uniform(1e-5, 1e-2)
    run = 0
    for lr in lrs:
        for beta1 in beta1s:
            for beta2 in beta2s:
                for batch_size in batch_sizes:
                    # lr = 0.00104
                    # beta1 = 0.9034
                    # beta2 = 0.997
                    # batch_size = 8
                    # weight_decay = 0.00226
                    print("Training parameters: lr: {}, betas: ({}, {}), batch size: {}, decay {}".format(lr, beta1, beta2, batch_size, weight_decay))
                    model = CustomResNet()
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                        lr=2e-4,
                                                        betas=(beta1, beta2),
                                                        weight_decay=weight_decay)
                    loss_fcn = torch.nn.CrossEntropyLoss()
                    solver = Solver(model, optimizer, loss_fcn)

                    # get training and test datasets from torch
                    # training_data = RotationDataSet("/home/peanut/Documents/cs231n_final_project/data/detection_bounding_boxes/train", n=None)
                    # test_data = RotationDataSet("/home/peanut/Documents/cs231n_final_project/data/detection_bounding_boxes/test", n=None)

                    training_data = RotationDataSet("/home/ubuntu/mrunal/cs231n_final_project/data/detection_bounding_boxes/train", n=None)
                    test_data = RotationDataSet("/home/ubuntu/mrunal/cs231n_final_project/data/detection_bounding_boxes/test", n=None)

                    # create dataloaders for datasets
                    train_dataloader = DataLoader(training_data, batch_size=int(batch_size), shuffle=False)
                    test_dataloader = DataLoader(test_data, batch_size=int(batch_size), shuffle=False)

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
                                save_every=None,
                                print_every=200,
                                verbose=True)

                    if best_val_acc is None or solver.best_val_acc > best_val_acc:
                        best_val_acc = solver.best_val_acc
                        best_model = "Training parameters: lr: {}, betas: ({}, {}), batch size: {}, decay {}".format(lr, beta1, beta2, batch_size, weight_decay)
                        best_model_epoch_idx = solver.best_val_epoch_idx

                        print("Found a good model")
                        print(best_model)
                        print(best_val_acc)
                        print(best_model_epoch_idx)

                    # train_acc =
                    # # Save results
                    # res = {"lr": lr, "beta1": beta1, "beta2": beta2, "decay": weight_decay, "train_acc": train_acc,
                    #         "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss}

    print("Done")
    print(best_model)
    print(best_val_acc)
    print(best_model_epoch_idx)
    solver.save_solver(os.path.join(SCRIPT_DIR,'checkpoints','mrunal_trained_model.pt'))

    # plot training/validation accuracy and loss
    # plot_loss_and_acc(solver.train_loss_history, solver.train_acc_history)
    # plot_loss_and_acc(solver.val_loss_history, solver.val_acc_history)

def train():
    # instantiate model, optimizer, and loss function for solver
    model = CustomResNet()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=2e-4,
                                        weight_decay=weight_decay)
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

if __name__ == "__main__":
    train_gridsearch()