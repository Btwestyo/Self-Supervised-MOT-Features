# vim: expandtab:ts=4:sw=4
import os
import sys
import errno
import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from scripts.models.model_puzzle import PuzzleNet
from scripts.models.model_pretrained import PreTrainedResNet

DATA_DIR = os.path.join(SCRIPT_DIR,'..','data')
MOT_DIR = os.path.join(DATA_DIR,'MOT17','train')

### CHANGE THESE ### 
OUTPUT_DIR = os.path.join(SCRIPT_DIR,'..','output','MOT17-train','resnet','features')
CKPT_DIR = os.path.join(SCRIPT_DIR,'..','scripts','training','checkpoints','mars_puzzle_1024_ckpt.pt')
MODEL_CLASS = "PreTrainedResNet"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
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

class CustomImageEncoder(object):

    def __init__(self, model_class, checkpoint_filename):

        self.model = torch.nn.Module()

        # create new model
        exec('self.model = '+model_class+'(checkpoint_filename)')

        # set model to evaluation mode
        self.model.eval()

        # input shape
        self.image_shape = [224, 224]

    def __call__(self, data_x, batch_size=32):
        data_x = torch.tensor(data_x).float()
        data_x = torch.permute(data_x, (0,3,1,2))
        with torch.no_grad():
            out = self.model(data_x.to(DEVICE)).cpu().numpy().astype(np.float32)
        return out

def create_box_encoder(model_class, model_filename, batch_size=32):
    image_encoder = CustomImageEncoder(model_class, model_filename)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)

        if len(image_patches.shape)<4:
            # if there are no valid patches, do not pass anything to net
            return None

        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None, detector='all'):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        if detector in sequence or detector=='all':
            print("Processing %s" % sequence)
            sequence_dir = os.path.join(mot_dir, sequence)

            image_dir = os.path.join(sequence_dir, "img1")
            image_filenames = {
                int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
                for f in os.listdir(image_dir)}

            detection_file = os.path.join(
                detection_dir, sequence, "det/det.txt")
            detections_in = np.loadtxt(detection_file, delimiter=',')
            detections_out = []

            frame_indices = detections_in[:, 0].astype(np.int64)
            min_frame_idx = frame_indices.astype(np.int64).min()
            max_frame_idx = frame_indices.astype(np.int64).max()
            for frame_idx in range(min_frame_idx, max_frame_idx):
                print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
                mask = frame_indices == frame_idx
                rows = detections_in[mask]

                if frame_idx not in image_filenames:
                    print("WARNING could not find image for frame %d" % frame_idx)
                    continue
                bgr_image = cv2.imread(
                    image_filenames[frame_idx], cv2.IMREAD_COLOR)
                features = encoder(bgr_image, rows[:, 2:6].copy())
                if features is not None:
                    detections_out += [np.r_[(row, feature)] for row, feature
                                    in zip(rows, features)]

            output_filename = os.path.join(output_dir, "%s.npy" % sequence)
            np.save(
                output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model_ckpt",
        default=CKPT_DIR,
        help="Path to pytorch model checkpoint dictionary containing 'model_state_dict' key")
    parser.add_argument(
        "--model_class",
        default=MODEL_CLASS,
        help="Exact name of pytorch model class in scripts/models folder.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        default=MOT_DIR)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default=OUTPUT_DIR)
    parser.add_argument(
        "--detection_dir", help="Detections directory."
        " exist.", default=None)
    parser.add_argument(
        "--detector", help="Detections to use for BBox images. Default: FRCNN",
         default='all')
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model_class, args.model_ckpt, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir, args.detector)


if __name__ == "__main__":
    main()
