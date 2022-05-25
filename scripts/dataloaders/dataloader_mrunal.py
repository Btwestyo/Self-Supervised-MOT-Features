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

from scripts.plotting.plot_results import plot_loss_and_acc

class RotationDataSet(Dataset):
  def __init__(self, dir, n=None):
      """
      Args:
          csv_file (string): Path to the csv file with annotations.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      super(RotationDataSet, self).__init__()
      self.transform = transforms.Compose([
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      self.root_dir = dir
      self.files = sorted(filter(lambda x: os.path.isfile(os.path.join(dir, x)),
                      os.listdir(dir)))
      if n is None:
        self.n = len(self.files)
      else:
        self.n = min(n, len (self.files))

      print("{} files in directory {}".format(len(self.files), dir))

  def __len__(self):
    return int(4*self.n)

  def __getitem__(self, idx_in):
    rot = idx_in % 4
    idx = int(idx_in / 4)

    try:
      # print("Index {}".format(idx_in))
      path = os.path.join(self.root_dir, str(self.files[idx]))
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