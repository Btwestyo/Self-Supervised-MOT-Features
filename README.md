# cs231n_final_project
# Data
- Download MOT17 dataset:
- Move data to /data
- Create a folder detection_bounding_boxes in /data
- Run ./scripts/generate_detections.bash to generate the bounding box images. Update the arguments in generate_detections.bash based on the directory names for your machine
- You should now have bounding box images in /data/name_of_bounding_box_dir
- The default code assumes that /data/detection_bounding_boxes/train and /data/detection_bounding_boxes/test exist. If you have different folder names you will need to update the directory names in the dataloader