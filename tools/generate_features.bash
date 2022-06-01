#!/bin/bash

# NOTE: YOU MUST EDIT THE GENERATE_FEATURES.PY IMPORTS
#       WITH THE MODEL YOU PLAN TO USE. IT WILL THROW AN ERROR
#       OTHERWISE.

# Arguments:

    # Set to the name of your pytorch model class
    #   model_class -   default: PuzzleNet

    # Set to directory containing your model checkpoint.
    # It is assumed that the checkpoint contains a dictionary with
    # with key 'model_state_dict' containing your model's state dict.
    #   model_ckpt  -   default: ./scripts/training/checkpoints/mars_puzzle_512_ckpt.pt

    #   mot_dir     -   default: ./data/MOT17/train

    # change the path to your model's folder in ./output/MOT17-train
    #   output_dir  -   default: ./output/MOT17-train/puzzle/data

    #   detector    -   default: 'all' {all, FRCNN, DPM, SDP}  

python3 tools/generate_features.py