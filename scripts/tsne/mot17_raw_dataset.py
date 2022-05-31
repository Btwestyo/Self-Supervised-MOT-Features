from os import path, listdir
import os
import cv2

import numpy as np
import torch
from torchvision import transforms
import random
from PIL import Image, ImageFile
import random

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Get colors map
random.seed(10)
colors_per_class = {}
for i in range(200):
    colors_per_class[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]


# processes Animals10 dataset: https://www.kaggle.com/alessiocorrado99/animals10
class MOT17Dataset(torch.utils.data.Dataset):
    def __init__(self, dir, num_images=1000):

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.root_dir = dir

        self.gt_file = os.path.join(dir, "gt", "gt.txt")
        self.detections_in = np.loadtxt(self.gt_file, delimiter=',')

        self.dir = os.path.join(dir, "img1")
        self.image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(self.dir, f)
            for f in os.listdir(self.dir)}

        self.patch_dir = "/home/peanut/Documents/cs231n_final_project/scripts/tsne/data/patches"

    def __len__(self):
        return self.detections_in.shape[0]

    def __getitem__(self, detection_index):
        image_idx = self.detections_in[detection_index, 0]
        # image = Image.open(self.image_filenames[image_idx])
        image = cv2.imread(self.image_filenames[image_idx], cv2.IMREAD_COLOR)

        # Extract image patch\
        image_shape = [224, 224]
        box = self.detections_in[detection_index, 2:6].copy()
        image_cv2 = self.extract_image_patch(image, box, image_shape[:2])
        image = Image.fromarray(image_cv2.astype('uint8'), 'RGB')

        image = self.transform(image) # some images in the dataset cannot be processed - we'll skip them
        patch_dir = os.path.join(self.patch_dir, "{}.jpeg".format(str(detection_index)))
        if not cv2.imwrite(patch_dir, image_cv2):
            print("Failed to write image")

        label = self.detections_in[detection_index, 1]
        dict_data = {
                'image' : image,
                'label' : label,
                'image_path' : patch_dir
        }
        return dict_data

    def extract_image_patch(self, image, bbox, patch_shape):
        """Extract image patch from bounding box.

        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.

        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.

        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int64)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, tuple(patch_shape[::-1]))
        return image


# Skips empty samples in a batch
def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    data = MOT17Dataset("/home/peanut/Documents/cs231n_final_project/data/MOT17/train/MOT17-02-FRCNN")
    for i in range(len(data)):
        d = data[i]

        print(d["label"])