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

#from scripts.plotting.plot_results import plot_loss_and_acc

class VanillaImageDataset(Dataset):
  def __init__(self, dir):
      """
      Args:
          csv_file (string): Path to the csv file with annotations.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      super(VanillaImageDataset, self).__init__()
      self.root_dir = dir
      self.files = sorted(filter(lambda x: os.path.isfile(os.path.join(dir, x)),
                      os.listdir(dir)))
      print("{} files in directory {}".format(len(self.files), dir))

  def __len__(self):
    return int(len(self.files))

  def __getitem__(self, idx_in):
    idx = int(idx_in)

    try:
      # print("Index {}".format(idx_in))
      path = os.path.join(self.root_dir, str(self.files[idx]))
      # print("loading image {}".format(path))
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      if(type(img) == type(None)): #.DS_Store can throw thigns off
        print(path)
        quit()
      else:
        img = cv2.resize(img, dsize=(112, 112), interpolation=cv2.INTER_CUBIC)
        img = img/255.

      img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
      img = torch.tensor(img).float()

    except Exception as e:
      print("Error {}".format(e))
      import pdb; pdb.set_trace()
      return None, None

    return img, img