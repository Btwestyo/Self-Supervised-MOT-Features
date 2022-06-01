#!/bin/bash

# Load data from mot_dir and save in output_dir
# Optional arguments:
#   mot_dir     -   default: ./data/MOT17
#   output_dir  -   default: ./data/detection_bounding_boxes
#   partition   -   default: 'both' (generate both train and test images)
#   detector    -   default: 'FRCNN' (FRCNN, DPM, or SDP)  

python3 tools/generate_bounding_boxes_parallel.py