import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter

'''设置数据集路径'''
root = '/home/xsh/XSH/MADNet-master/NUPT-Metro'

nupt_train = os.path.join(root, 'train_data', 'images')
nupt_test = os.path.join(root, 'test_data', 'images')

path_sets = [nupt_train, nupt_test]

if not os.path.exists(nupt_train.replace('images', 'gt_fidt_map')):
    os.makedirs(nupt_train.replace('images', 'gt_fidt_map'))

if not os.path.exists(nupt_test.replace('images', 'gt_fidt_map')):
    os.makedirs(nupt_test.replace('images', 'gt_fidt_map'))

if not os.path.exists(nupt_train.replace('images', 'gt_show')):
    os.makedirs(nupt_train.replace('images', 'gt_show'))

if not os.path.exists(nupt_test.replace('images', 'gt_show')):
    os.makedirs(nupt_test.replace('images', 'gt_show'))

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

def fidt_generate1(im_data, gt_data, lamda):
    size = im_data.shape
    new_im_data = cv2.resize(im_data, (lamda * size[1], lamda * size[0]), 0)

    new_size = new_im_data.shape
    d_map = (np.zeros([new_size[0], new_size[1]]) + 255).astype(np.uint8)
    gt = lamda * gt_data

    for o in range(0, len(gt)):
        x = np.max([1, math.floor(gt[o][1])])
        y = np.max([1, math.floor(gt[o][0])])
        if x >= new_size[0] or y >= new_size[1]:
            continue
        d_map[x][y] = d_map[x][y] - 255

    distance_map = cv2.distanceTransform(d_map, cv2.DIST_L2, 0)
    distance_map = torch.from_numpy(distance_map)
    distance_map = 1 / (1 + torch.pow(distance_map, 0.02 * distance_map + 0.75))
    distance_map = distance_map.numpy()
    distance_map[distance_map < 1e-2] = 0

    return distance_map

for img_path in img_paths:
    print(img_path)
    
    # 读取图片
    Img_data = cv2.imread(img_path)
    
    # 获取对应的 YOLO 标注文件路径
    txt_path = img_path.replace('images', 'ground_truth').replace('.jpg', '.txt')
    
    # 读取 YOLO 标注文件
    with open(txt_path, 'r') as txt_file:
        yolo_data = txt_file.readlines()
    
    # 初始化 Gt_data 用于存储点坐标
    Gt_data = []
    
    # 获取图片尺寸
    img_height, img_width = Img_data.shape[:2]
    
    # 遍历每个 YOLO 标注点
    for data in yolo_data:
        parts = data.strip().split()
        if len(parts) >= 5:
            # 提取中心点坐标（归一化）
            x_center = float(parts[1])
            y_center = float(parts[2])
            
            # 转换为实际坐标
            x = x_center * img_width
            y = y_center * img_height
            
            # 添加到 Gt_data
            Gt_data.append([x, y])
    
    # 转换为 numpy 数组
    Gt_data = np.array(Gt_data)
    

    fidt_map1 = fidt_generate1(Img_data, Gt_data, 1)
    
    # 创建 kpoint 图
    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for i in range(len(Gt_data)):
        x = int(round(Gt_data[i][0]))
        y = int(round(Gt_data[i][1]))
        if 0 <= y < Img_data.shape[0] and 0 <= x < Img_data.shape[1]:
            kpoint[y, x] = 1
    
    # 保存到 .h5 文件
    h5_path = img_path.replace('images', 'gt_fidt_map').replace('.jpg', '.h5')
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, 'w') as hf:
        hf['fidt_map'] = fidt_map1
        hf['kpoint'] = kpoint

    fidt_map1 = fidt_map1
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    
    # 显示或保存可视化结果
    cv2.imwrite(img_path.replace('images', 'gt_show').replace('jpg', 'jpg'), fidt_map1)