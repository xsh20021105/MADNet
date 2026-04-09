import os
import time
import glob
import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import math
import scipy.io as io
from matplotlib import pyplot as plt
import sys

root = '/home/xsh/XSH/MADNet-master/NUPT-Metro'

nupt_test = os.path.join(root, 'test_data', 'images')
path_sets = [nupt_test]

if not os.path.exists(nupt_test):
    sys.exit("The path is wrong, please check the dataset path.")


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

f = open('./nupt_gt.txt', 'w+')
k = 1

for img_path in img_paths:
    print(img_path)
    
    txt_path = img_path.replace('images', 'ground_truth').replace('.jpg', '.txt')
    
    with open(txt_path, 'r') as txt_file:
        yolo_data = txt_file.readlines()
    
    total_count = len(yolo_data)
    
    f.write(f"{k} {total_count} ")
    
    for data in yolo_data:
        parts = data.strip().split()
        if len(parts) >= 5:
            x_center = float(parts[1])
            y_center = float(parts[2])
            
            f.write(f"{x_center:.6f} {y_center:.6f} 4 8 ")
    
    f.write("\n")
    k += 1

f.close()

