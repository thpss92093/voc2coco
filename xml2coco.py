#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid

import numpy as np
import PIL.Image

import labelme
import xml.etree.ElementTree as XET

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
    print('Creating dataset:', args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    # read label class name(class_name) from labels.txt
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        # print("class_name: ", class_name)
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(args.output_dir, 'annotations.json')
    label_files = glob.glob(osp.join(args.input_dir, '*.xml'))

    for image_id, label_file in enumerate(label_files):
        print('Generating dataset from:', label_file)
        tree = XET.parse(label_file)
        root = tree.getroot()

        filename = get_and_check(root, 'filename', 1).text                  # <filename> to {images: file_name}
        # folder = get_and_check(root, 'folder', 1).text                    # <filename> not be used
        # imagesize = get_and_check(root, 'imagesize', 1)                   # <imagesize> not be used
        # nrows = int(get_and_check(imagesize, 'nrows', 1).text)            # <imagesize> <nrows> not be used
        # ncols = int(get_and_check(imagesize, 'ncols', 1).text)            # <imagesize> <ncols> not be used
        objectname = get(root, 'object')                                    # <object> contain label info, like classname, point

        out_img_file = osp.join(args.output_dir, 'JPEGImages', filename)
        img_file = osp.join(osp.dirname(label_file), filename)
        img = np.asarray(PIL.Image.open(img_file).convert('RGB'))
        PIL.Image.fromarray(img).save(out_img_file)

        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],                                            # height=nrows,
            width=img.shape[1],                                             # width=ncols,
            date_captured=None,
            id=image_id,
        ))

        masks = {}                                                          # for {annotations: area}
        segmentations = collections.defaultdict(list)                       # for {annotations: segmentation}

        for obj in objectname:
            label_name = get_and_check(obj, 'name', 1).text                 # <object> <name> get the label class
            shape_type = 'polygon'                                          # for mask shape_to_mask
            polygon = get_and_check(obj, 'polygon', 1)                      # <object> <polygon>
            pts = get(polygon, 'pt')                                        # <object> <polygon> <pt>
            points = []
            for pt in pts:
                pt_x = float(get_and_check(pt, 'x', 1).text)                # <object> <polygon> <pt> <x>
                pt_y = float(get_and_check(pt, 'y', 1).text)                # <object> <polygon> <pt> <y>
                points.append([pt_x, pt_y])

            group_id = uuid.uuid1()
            instance = (label_name, group_id)
            mask = labelme.utils.shape_to_mask(img.shape[:2], points, shape_type)
            points = np.asarray(points).flatten().tolist()
            masks[instance] = mask
            segmentations[instance].append(points)                          # label points convert <object> <polygon> <pt> to {annotations: segmentation}
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]
            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data['annotations'].append(dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=segmentations[instance],
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
