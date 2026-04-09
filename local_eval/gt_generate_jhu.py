import os
import glob
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import sys

'''please set your dataset path'''
root = '/home/xsh/XSH/MADNet-master/jhu_crowd_v2.0'

jhu_test_images = os.path.join(root, 'test', 'images')
jhu_test_gt = os.path.join(root, 'test', 'gt')

if not os.path.exists(jhu_test_images) or not os.path.exists(jhu_test_gt):
    sys.exit("The path is wrong, please check the dataset path.")

img_paths = []
gt_paths = []

for img_path in glob.glob(os.path.join(jhu_test_images, '*.jpg')):
    img_paths.append(img_path)
    gt_file_name = os.path.basename(img_path).replace('.jpg', '.txt')
    gt_paths.append(os.path.join(jhu_test_gt, gt_file_name))

img_paths.sort()
gt_paths.sort()

f = open('./jhu_gt.txt', 'w+')
k = 1
for img_path, gt_path in zip(img_paths, gt_paths):

    print(img_path)
    print(gt_path)

    with open(gt_path, 'r') as gt_file:
        Gt_data = []
        for line in gt_file:
            parts = line.strip().split()
            if len(parts) >= 2:
                x, y = map(float, parts[:2])
                Gt_data.append([x, y])

    Gt_data = np.array(Gt_data)
    f.write('{} {} '.format(k, len(Gt_data)))

    for data in Gt_data:
        sigma_s = 4
        sigma_l = 8
        f.write('{} {} {} {} {} '.format(math.floor(data[0]), math.floor(data[1]), sigma_s, sigma_l, 1))
    f.write('\n')

    k = k + 1
f.close()



