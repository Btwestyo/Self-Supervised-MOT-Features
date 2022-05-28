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

class MarsDataSetPuzzle(Dataset):
  def __init__(self, dir, overfit=False):
      """
      Args:
          dir (string): Directory with all the images.
      """
      super(MarsDataSetPuzzle, self).__init__()
      self.transform = transforms.Compose([
                        transforms.Normalize(
                          mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])]
                        )
                        
      self.root_dir = dir
      self.files = []
      for subdir in os.listdir(dir):
        self.files.extend(sorted(map(lambda x: os.path.join(dir, subdir, x),
                        os.listdir(os.path.join(dir,subdir)))))

      if overfit and ('test' not in dir):
        # to check efficacy, overfit on small sample set
        self.files = np.random.choice(self.files,100,replace=False)
      elif ('test' not in dir):
        self.files = np.random.choice(self.files,25600,replace=False)
      else:
        self.files = np.random.choice(self.files,3200,replace=False)



      print("{} files in directory {}".format(len(self.files), dir))

  def __len__(self):
    return int(len(self.files))

  def __getitem__(self, idx_in):
    patch = idx_in % 4
    idx   = int(idx_in / 4)

    img = cv2.imread(str(self.files[idx]), cv2.IMREAD_COLOR)

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
    training_data = MarsDataSetPuzzle(
      os.path.join(SCRIPT_DIR,'..','..','data','mars','bbox_train'))

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    for i in range(20):
        img, gt = training_data.__getitem__(i)
        print(gt)