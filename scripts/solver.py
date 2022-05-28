import time
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

"""Class for training, validating, and testing PyTorch models.
Requires setup for model, optimizer, etc. outside of this class."""
class Solver(object):

    def __init__(self,
                 model:torch.nn.Module,
                 optimizer:torch.optim.Optimizer,
                 loss_fcn:torch.nn.Module) -> None:
        """Initialize the solver with a module, optimizer, and
        loss function."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fcn = loss_fcn
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.train_loss_history = None
        self.val_loss_history = None

        self.train_acc_history = None
        self.val_acc_history = None

    def train(self,
        epochs:int,
        train_data_loader:DataLoader,
        val_data_loader:DataLoader,
        save_path:str=None,
        start_epoch:int=0,
        save_every:int=1,
        print_every:int=100,
        verbose=False) -> None:

        # Read in the dataloader if a filename is given
        if isinstance(train_data_loader, str):
            train_data_loader = torch.load(train_data_loader)

        if isinstance(val_data_loader, str):
            val_data_loader = torch.load(val_data_loader)

        num_trainloader = len(train_data_loader) # number of batches
        num_valloader   = len(val_data_loader) # number of batches
        batch_size      = train_data_loader.batch_size # batch size

        # pre-allocate storage for loss and accuracy values
        epoch_loss  = [np.zeros(num_trainloader).copy() for _ in range(epochs)]
        epoch_acc   = [np.zeros(num_trainloader).copy() for _ in range(epochs)]
        val_loss    = [np.zeros(num_valloader).copy() for _ in range(epochs)]
        val_acc     = [np.zeros(num_valloader).copy() for _ in range(epochs)]

        if verbose:
            print('Start Training From Epoch #{:d}'.format(start_epoch+1))
            print('Model sent to device: ', self.DEVICE)

        self.model = self.model.to(self.DEVICE) # send model to GPU if available

        # Generate ID string for checkpoint files
        start_time = time.time()
        id_str = 'train_'+str(int(start_time))

        # set model to training mode
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(start_epoch, epochs):
            if verbose:
                print("----- TRAINING EPOCH #{:d} -----".format(1+epoch))

            # pre-allocate storage for batch loss
            batch_loss = np.zeros(num_trainloader)
            batch_acc  = np.zeros(num_trainloader)

            for batch_idx, batch in tqdm(enumerate(train_data_loader)):
                nn_input, target = batch # input and ground truth lables
                nn_input = nn_input.to(self.DEVICE) #image
                target = target.to(self.DEVICE)

                nn_preds = self.model(nn_input) # inference
                loss     = self.loss_fcn(nn_preds, target) # loss
                acc      = self.model.compute_accuracy(nn_preds, target)

                loss.backward()  # compute the gradients
                self.optimizer.step() # update the model parameters

                # ensure that loss is detatched from computational graph
                # before storing
                batch_loss[batch_idx] = float(loss.detach())
                batch_acc[batch_idx] = float(acc.detach())

                # print intermediate results
                if verbose and ((batch_idx+1) % print_every == 0):
                    self._print_stats(batch_idx, num_trainloader, batch_loss, batch_acc)

                # set gradients to None
                self.optimizer.zero_grad(set_to_none=False)

            # save stats for this epoch
            epoch_loss[epoch] = batch_loss
            epoch_acc[epoch] = batch_acc

            # get validation stats for this epoch
            val_loss[epoch], val_acc[epoch] = self.validate(val_data_loader)
            if verbose:
                print("Validation Loss: ", val_loss[epoch].mean(),
                      " Acc: ", val_acc[epoch].mean())

            # store all values in solver
            self.train_loss_history = epoch_loss
            self.train_acc_history  = epoch_acc
            self.val_loss_history   = val_loss
            self.val_acc_history    = val_acc

            if (save_path is not None) and ((epoch+1)%save_every==0):
                path = join(save_path, id_str+'_'+str(epoch)+'.pt')
                self.save_checkpoint(path, epoch)

        end_time = time.time()

        if verbose:
            print("---------- Training Finished ----------")
            print("Total training time: ", end_time-start_time)

    def validate(self, val_data_loader) -> None:
        num_valloader   = len(val_data_loader) # number of batches

        # re-instantiate model
        model = copy.deepcopy(self.model)
        model = self.model.to(self.DEVICE) # send model to GPU if available

        batch_loss = np.zeros(num_valloader)
        batch_acc  = np.zeros(num_valloader)

        # set model to evaluation mode
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data_loader):

                nn_input, target = batch # input and ground truth labels

                nn_input = nn_input.to(self.DEVICE)
                target = target.to(self.DEVICE)

                nn_preds = self.model(nn_input) # inference

                batch_loss[batch_idx] = self.loss_fcn(nn_preds, target) # loss
                batch_acc[batch_idx]  = self.model.compute_accuracy(nn_preds, target)

        return batch_loss, batch_acc

    def test(self, test_data_loader) -> None:
        num_testloader   = len(test_data_loader) # number of batches

        # re-instantiate model
        model = copy.deepcopy(self.model)
        model = self.model.to(self.DEVICE) # send model to GPU if available

        batch_loss = np.zeros(num_testloader)
        batch_acc  = np.zeros(num_testloader)

        # set model to evaluation mode
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data_loader):

                nn_input, target = batch # input and ground truth lables

                nn_input = nn_input.to(self.DEVICE)
                target = target.to(self.DEVICE)

                nn_preds = self.model(nn_input) # inference

                batch_loss[batch_idx] = self.loss_fcn(nn_preds, target) # loss
                batch_acc[batch_idx]  = self.model.compute_accuracy(nn_preds, target)

        return batch_loss, batch_acc

    def save_solver(self, save_path):
        torch.save({'solver':self}, save_path)

    def save_checkpoint(self, save_path, epoch):
        torch.save({'epoch':epoch,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),
                    'train_loss':self.train_loss_history,
                    'train_acc':self.train_acc_history,
                    'val_loss':self.val_loss_history,
                    'val_acc':self.val_acc_history
                    }, save_path)

    def _print_stats(self, batch_idx, num_trainloader, batch_loss, batch_acc):
        print(
            "Iteration: #{:d}/{:d} \tLoss(curr/avg) {:.4f}/{:.4f}\t Acc(curr/avg) {:.4f}/{:.4f}"
            .format(batch_idx+1,
                    num_trainloader,
                    batch_loss[batch_idx],
                    np.mean(batch_loss[:(batch_idx+1)]),
                    batch_acc[batch_idx],
                    np.mean(batch_acc[:(batch_idx+1)])
                    )
            )
