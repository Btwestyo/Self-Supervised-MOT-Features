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
from torchvision import transforms, utils

# unfortunately this is required to use relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

from scripts.models.model_example import ConvClassifier
from scripts.solver_rotation import Solver
from scripts.plotting.plot_results import plot_loss_and_acc
from scripts.models.resnet import CustomResNet


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

    def __init__(self, type):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(RotationDataSet, self).__init__()
        self.root_dir = "/home/peanut/Documents/cs231n_final_project/data/detection_bb"
        self.transform = transforms.Compose([
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if type == "train":
          self.zero_idx = 0
        else:
          self.zero_idx = int(len(os.listdir(self.root_dir))/2) - 1

    def __len__(self):
      return int(4*len(os.listdir(self.root_dir))/2)

    def __getitem__(self, idx_in):
      rot = idx_in % 4
      idx = self.zero_idx + int(idx_in / 4)

      try:
        # print("Index {}".format(idx_in))
        path = self.root_dir + "/image_" + str(idx) + ".jpeg"
        # print("loading image {}".format(path))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

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

        img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
        img = torch.tensor(img).float()
        img = self.transform(img)

      except Exception as e:
        print("Error {}".format(e))
        import pdb; pdb.set_trace()
        return None, None

      return img, rot