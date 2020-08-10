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

import csv
import shutil

output_path = "/home/lily/voc2coco/train/"
input_dir = "/home/lily/voc2coco/delta_5"

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
    parser.add_argument('csv', help='csv file')
    args = parser.parse_args()

    i = 0
    with open(args.csv, mode='r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            xml_file = row[0].replace("images", "annotations").replace("jpg", "xml")
            _xml_f = osp.join(input_dir, xml_file)
            _img_f = osp.join(input_dir, row[0])
            i += 1
            if "d435_1" in xml_file:
                output_xmlfile_name = output_path + "d1_" + xml_file[-14:]
                output_imgfile_name = output_path + "d1_" + _img_f[-14:]
            elif "d435_2" in xml_file:
                output_xmlfile_name = output_path + "d2_" + xml_file[-14:]
                output_imgfile_name = output_path + "d2_" + _img_f[-14:]
            elif "zedmini_2" in xml_file:
                output_xmlfile_name = output_path + "ze_" + xml_file[-14:]
                output_imgfile_name = output_path + "ze_" + _img_f[-14:]

            print(_xml_f)
            print(output_xmlfile_name)
            print(_img_f)
            print(output_imgfile_name)
            print("===========================")

            shutil.copyfile(_xml_f, output_xmlfile_name)
            shutil.copyfile(_img_f, output_imgfile_name)

        
        csvfile.close()
    print("len(label_files): ")
    print(i)


if __name__ == '__main__':
    main()