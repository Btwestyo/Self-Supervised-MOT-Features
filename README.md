# cs231n_final_project
# Data
- Download MOT17 dataset:
- Move data to /data
- Create a folder detection_bounding_boxes in /data
- Run ./tools/generate_detections.bash or ./tools/generate_detections_parallel.bash to generate the bounding box images. Update the arguments in generate_detections.bash based on the directory names for your machine
- You should now have bounding box images in /data/name_of_bounding_box_dir
- The default code assumes that the /data/detection_bounding_boxes directory exists.
