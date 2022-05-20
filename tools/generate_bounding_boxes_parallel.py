import os
import sys
import errno
import argparse
import numpy as np
import cv2
import multiprocessing

"""This script is an adaptation of the DeepSORT implementation for 
obtaining the bounding box crops from the MOT dataset. This version
runs with less overhead -- no tensorflow model is required. This script
uses multithreading to parallelize bounding box image saving."""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(
                os.path.dirname(SCRIPT_DIR)))

DATA_DIR = os.path.join(SCRIPT_DIR,'..','data')
MOT_DIR = os.path.join(DATA_DIR,'MOT17')
OUTPUT_DIR = os.path.join(SCRIPT_DIR,'..','data','detection_bounding_boxes')
    
# specifies which of the three provied provided detections should be
# used in generating the bounding box crops
DETECTOR = 'FRCNN'

def extract_image_patch(image, bbox):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
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
    return image

def generate_detections(mot_dir=MOT_DIR, output_dir=OUTPUT_DIR, detector=DETECTOR):
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

    """
    # raise error if output directory cannot be created
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        # only write detections from selected detector
        if detector in sequence:
            print("Processing %s" % sequence)
            p = multiprocessing.Process(target=threading_fcn, args = [sequence, mot_dir, output_dir])
            p.start()

def threading_fcn(sequence, mot_dir, output_dir):
    count = 0
    sequence_dir = os.path.join(mot_dir, sequence)

    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    detection_file = os.path.join(
        mot_dir, sequence, "det/det.txt")
    detections_in = np.loadtxt(detection_file, delimiter=',')

    frame_indices = detections_in[:, 0].astype(np.int64)
    min_frame_idx = frame_indices.astype(np.int64).min()
    max_frame_idx = frame_indices.astype(np.int64).max()

    for i in range(min_frame_idx, max_frame_idx):
        frame_idx = i #random.randint(min_frame_idx, max_frame_idx)
        # print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]

        if frame_idx not in image_filenames:
            print("WARNING could not find image for frame %d" % frame_idx)
            continue
        bgr_image = cv2.imread(
            image_filenames[frame_idx], cv2.IMREAD_COLOR)

        boxes = rows[:, 2:6].copy()
        for box in boxes:
            patch = extract_image_patch(bgr_image, box)

            # Save image patch
            image_path = os.path.join(output_dir, sequence + '_' + str(count) + ".jpeg")
            
            if not cv2.imwrite(image_path, patch):
                print("Failed to write imgae")

            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))

            count += 1


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Save BBox Image Patches")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        default=MOT_DIR)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default=OUTPUT_DIR)
    parser.add_argument(
        "--detector", help="Detections to use for BBox images. Default: FRCNN",
         default=DETECTOR)
    parser.add_argument(
        "--partition", help="Data partition of source images: train/test",
         default='both')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    if args.partition == 'both':
        partitions = ['train', 'test']
    else:
        partitions = [args.partition]
        
    for p in partitions:
        print('Generating %s images using %s detections' % (p, DETECTOR))
        generate_detections(mot_dir=os.path.join(args.mot_dir,p),
                            output_dir=os.path.join(args.output_dir,p),
                            detector=args.detector)