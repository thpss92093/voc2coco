# voc2coco
Convert labelme(MIT website version) xml file to coco dataset format (json file).

This code is based on https://github.com/wkentaro/labelme/blob/master/examples/instance_segmentation/labelme2coco.py which is used to convert labelme format json file to coco dataset format.

```bash
python2 xml2coco.py input_folder_path output_folder_path --labels labels.txt
```
Input:
  - input_folder_path: the image and label file(.xml) should in this folder
  - labels.txt: the label class name

Output:
  - output_folder_path/JPEGImages
  - output_folder_path/annotations.json
