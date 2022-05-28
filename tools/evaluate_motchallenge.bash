#!/bin/bash

# Arguments:
#     parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
#     parser.add_argument(
#         "--mot_dir", help="Path to MOTChallenge directory (train or test)",
#         default=MOT_DIR)
#     parser.add_argument(
#         "--detection_dir", help="Path to detections.",
#         default=DET_DIR)
#     parser.add_argument(
#         "--output_dir", help="Folder in which the results will be stored. Will "
#         "be created if it does not exist.", 
#         default=OUTPUT_DIR)
#     parser.add_argument(
#         "--min_confidence", help="Detection confidence threshold. Disregard "
#         "all detections that have a confidence lower than this value.",
#         default=0.0, type=float)
#     parser.add_argument(
#         "--min_detection_height", help="Threshold on the detection bounding "
#         "box height. Detections with height smaller than this value are "
#         "disregarded", default=0, type=int)
#     parser.add_argument(
#         "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
#         "detection overlap.", default=1.0, type=float)
#     parser.add_argument(
#         "--max_cosine_distance", help="Gating threshold for cosine distance "
#         "metric (object appearance).", type=float, default=0.2)
#     parser.add_argument(
#         "--nn_budget", help="Maximum size of the appearance descriptors "
#         "gallery. If None, no budget is enforced.", type=int, default=100)
#     return parser.parse_args()

python3 ./deep_sort/evaluate_motchallenge.py