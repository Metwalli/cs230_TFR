#!/usr/bin/env python
from glob import glob
import os
import cv2
pngs = glob('C:\data\cifar\\train/*.png')
data_dir = "C:\data\cifar\\train"
output_dir = "C:\data\cifar10\\train"
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck]']
for c in classes:
    if not os.path.exists(os.path.join(output_dir, c)):
        os.mkdir(os.path.join(output_dir, c))
    class_path = os.path.join(output_dir, c)
    for j in pngs:
        if c in j:
            img = cv2.imread(j)
            cv2.imwrite(os.path.join(output_dir, c, j.split(os.path.sep)[-1].split('.')[0]+ '.jpg'), img)
