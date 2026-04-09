# --------------------------------------------
# 公开的数据集——上海科技大学A转换为npy数据
# --------------------------------------------
import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''
shanghai_root = '/home/xsh/XSH/MADNet-master/ShanghaiTech'
qnrf_root = '/home/xsh/XSH/MADNet-master/UCF-QNRF_ECCV18'
jhu_root = '/home/xsh/XSH/MADNet-master/jhu_crowd_v2.0'

try:

    shanghaiAtrain_path = shanghai_root + '/part_A_final/train_data/images/'
    shanghaiAtest_path = shanghai_root + '/part_A_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("Generate ShanghaiA image list successfully")
except:
    print("The ShanghaiA dataset path is wrong. Please check you path.")

try:
    shanghaiBtrain_path = shanghai_root + '/part_B_final/train_data/images/'
    shanghaiBtest_path = shanghai_root + '/part_B_final/test_data/images/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully")
except:
    print("The ShanghaiB dataset path is wrong. Please check your path.")

try:
    Qnrf_train_path = qnrf_root + '/train_data/images/'
    Qnrf_test_path = qnrf_root + '/test_data/images/'

    train_list = []
    for filename in os.listdir(Qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Qnrf_train_path + filename)
    train_list.sort()
    np.save('./npydata/qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(Qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(Qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/qnrf_test.npy', test_list)
    print("Generate QNRF image list successfully")
except:
    print("The QNRF dataset path is wrong. Please check your path.")

try:

    Jhu_train_path = jhu_root + '/train/images_2048/'
    Jhu_val_path = jhu_root + '/val/images_2048/'
    jhu_test_path = jhu_root + '/test/images_2048/'

    train_list = []
    for filename in os.listdir(Jhu_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Jhu_train_path + filename)
    train_list.sort()
    np.save('./npydata/jhu_train.npy', train_list)

    val_list = []
    for filename in os.listdir(Jhu_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(Jhu_val_path + filename)
    val_list.sort()
    np.save('./npydata/jhu_val.npy', val_list)

    test_list = []
    for filename in os.listdir(jhu_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(jhu_test_path + filename)
    test_list.sort()
    np.save('./npydata/jhu_test.npy', test_list)

    print("Generate JHU image list successfully")
except:
    print("The JHU dataset path is wrong. Please check your path.")
