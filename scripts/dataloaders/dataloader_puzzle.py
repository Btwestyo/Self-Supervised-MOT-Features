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

class PuzzleDataSet(Dataset):
  def __init__(self, dir):
      """
      Args:
          dir (string): Directory with all the images.
      """
      super(PuzzleDataSet, self).__init__()
      self.transform = transforms.Compose([
                        transforms.Normalize(
                          mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])]
                        )
                        
      self.root_dir = dir
      self.files = sorted(filter(lambda x: os.path.isfile(os.path.join(dir, x)),
                      os.listdir(dir)))
      # print("{} files in directory {}".format(len(self.files), dir))

  def __len__(self):
    return int(4*len(self.files))

  def __getitem__(self, idx_in):
    patch = idx_in % 4
    idx   = int(idx_in / 4)

    # print("Index {}".format(idx_in))
    path = os.path.join(self.root_dir, str(self.files[idx]))
    # print("loading image {}".format(path))
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    h,w,_ = img.shape

    if patch == 0:
      img = img[:int(h/2),:int(w/2),:]
    elif patch == 1:
      img = img[:int(h/2),int(w/2):int(w),:]
    elif patch == 2:
      img = img[int(h/2):int(h),:int(w/2),:]
    elif patch == 3:
      img = img[int(h/2):int(h),int(w/2):int(w),:]
    else:
      print("ERROR")

    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
    img = torch.tensor(img).float()
    img = self.transform(img)

    return img, patch

  
if __name__ == '__main__':
    training_data = PuzzleDataSet(
      os.path.join(SCRIPT_DIR,'..','..','data','detection_bounding_boxes','train'))

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    for i in range(20):
        training_data.__getitem__(i)