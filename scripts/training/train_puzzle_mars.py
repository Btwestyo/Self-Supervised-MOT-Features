from numpy import Inf
import torch
from torch.utils.data import DataLoader
import os
import sys

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.models.model_convresnet import CustomResNetConv
from scripts.solver import Solver
from scripts.plotting.plot_results import plot_loss_and_acc
from scripts.dataloaders.dataloader_mars_puzzle import MarsDataSetPuzzle

if __name__ == "__main__":
    # instantiate model, optimizer, and loss function for solver
    model = CustomResNetConv()
    optimizer = torch.optim.Adam(lr=1e-5, params=model.parameters())
    loss_fcn = torch.nn.CrossEntropyLoss()

    solver = Solver(model, optimizer, loss_fcn)

    # get training and test datasets from torch
    training_data =  MarsDataSetPuzzle(
        os.path.join(SCRIPT_DIR,'..','..','data','mars','bbox_train'),
        overfit=False)
    
    test_data = MarsDataSetPuzzle(
        os.path.join(SCRIPT_DIR,'..','..','data','mars','bbox_test'),
        overfit=False)

    # create dataloaders for datasets
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

    # train the model
    solver.train(epochs=10, 
                train_data_loader=train_dataloader,
                val_data_loader=test_dataloader,
                save_path=os.path.join(SCRIPT_DIR,'checkpoints'),
                save_every=1,
                print_every=10,
                verbose=True)

    # plot training/validation accuracy and loss
    plot_loss_and_acc(solver.train_loss_history, solver.train_acc_history)
    plot_loss_and_acc(solver.val_loss_history, solver.val_acc_history)

    # save final solver object
    solver.save_solver(os.path.join(SCRIPT_DIR,'checkpoints','mars_puzzle_trained_model.pt'))