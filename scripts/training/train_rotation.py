import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.models as models
from matplotlib import pyplot as plt

import os
import sys
import cv2

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.models.model_example import ConvClassifier
from scripts.solver import Solver
from scripts.plotting.plot_results import plot_loss_and_acc

# class RotationDataSet(Dataset):

#     def __init__(self):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         super(RotationDataSet, self).__init__()
#         self.root_dir = "/home/peanut/Documents/cs231n_final_project/data/detection_bb"


#     def __len__(self):
#       return len(os.listdir(self.root_dir))

#     def __getitem__(self, idx):
#       path = self.root_dir + "/image_" + str(idx) + ".jpeg"
#       img = cv2.imread(path, cv2.IMREAD_COLOR)
#       img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
#       img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#       img180 = cv2.rotate(img, cv2.ROTATE_180)
#       img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

#       return {
#                 "data": [img, img90, img180, img270],
#                 "label": [0, 1, 2, 3]
#               }

class RotationDataSet(Dataset):

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(RotationDataSet, self).__init__()
        self.root_dir = "/home/peanut/Documents/cs231n_final_project/data/detection_bb"


    def __len__(self):
      return 4*len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
      rot = idx % 4
      idx = int(idx / 4)

      path = self.root_dir + "/image_" + str(idx) + ".jpeg"
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)

      if rot == 0:
        pass
      elif rot == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
      elif rot == 2:
        img = cv2.rotate(img, cv2.ROTATE_180)
      elif rot == 3:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
      else:
        print("ERROR")

      return img, rot

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pretrained =  models.resnet18(pretrained=True)
        self.pretrained = nn.Sequential(*list(self.pretrained.modules())[:-1])
        self.custom_layers = nn.Sequential(nn.Linear(512, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 256),
                                           nn.Softmax())

        for param in self.pretrained.parameters():
          self.pretrained.requires_grad = False

    def forward(self, x):
        x = self.pretrained(x)
        x = self.custom_layers(x)
        return x

if __name__ == "__main__":
    # instantiate model, optimizer, and loss function for solver
    # model = ConvClassifier(in_ch=1, base_filt=8, dim_reduce=2)
    model = MyModel()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss_fcn = torch.nn.CrossEntropyLoss()

    solver = Solver(model, optimizer, loss_fcn)

    # get training and test datasets from torch
    training_data = RotationDataSet()
    test_data = RotationDataSet()

    # create dataloaders for datasets
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Viz a sample
    # for sample in train_dataloader:
    #   for i in range(64):
    #     plt.imshow(sample[0][i, :, :, :], interpolation='nearest')
    #     plt.show()
    #     import pdb; pdb.set_trace()

    # train the model
    solver.train(epochs=10,
                train_data_loader=train_dataloader,
                val_data_loader=test_dataloader,
                save_path=None,#os.path.join(SCRIPT_DIR,'checkpoints'),
                save_every=1,
                print_every=32,
                verbose=True)

    # plot training/validation accuracy and loss
    plot_loss_and_acc(solver.train_loss_history, solver.train_acc_history)
    plot_loss_and_acc(solver.val_loss_history, solver.val_acc_history)

    # save final solver object
    solver.save_solver(os.path.join(SCRIPT_DIR,'checkpoints','example_trained_model.pt'))