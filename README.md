# Environment

	python >=3.6 
	pytorch >=1.4
	opencv-python >=4.0
	scipy >=1.4.0
	h5py >=2.10
	pillow >=7.0.0
	imageio >=1.18
	nni >=2.0 (python3 -m pip install --upgrade nni)

# Datasets

The NJUPT-Metro dataset is collected by ourselves. Please contact the corresponding author if you need access to it. All other datasets are publicly available on their official websites.

# Generate FIDT Ground-Truth

```
cd data
run  python fidt_generate_xx.py
```

“xx” means the dataset name, including sh, jhu, qnrf, and nwpu. You should change the dataset path.

# Model

Download the pretrained model from [Baidu-Disk](https://pan.baidu.com/s/1SaPppYrkqdWeHueNlcvUJw), passward:gqqm, or [OneDrive](https://1drv.ms/u/s!Ak_WZsh5Fl0lhCneubkIv1mTllAZ?e=0zMHSM)

# Quickly test
```
Download Dataset and Model  
Generate FIDT map ground-truth  
Generate image file list: python make_npydata.py
```

**Test example:**
```
python /home/xsh/XSH/MADNet-master/MADNet_test.py --dataset NUPT-Metro  --pre /home/xsh/XSH/MADNet-master/save_file/NUPT-Metro/model_best.pth --gpu_id 0 --visual True
```

**If you want to generate bounding boxes,**
```
python /home/xsh/XSH/MADNet-master/MADNet_test.py --dataset NUPT-Metro  --pre /home/xsh/XSH/MADNet-master/save_file/NUPT-Metro/model_best.pth  --visual True
(remember to change the dataset path in test.py)  
```

**Tips**:  
The GT format is:

```
1 total_count x1 y1 4 8 x2 y2 4 8 ..... 
2 total_count x1 y1 4 8 x2 y2 4 8 .....
```
The predicted format is:
```
1 total_count x1 y1 x2 y2.....
2 total_count x1 y1 x2 y2.....
```

**Train example:**

```
python /home/xsh/XSH/MADNet-master/MADNet_train.py --dataset NUPT-Metro --crop_size 256 --save_path ./save_file/NUPT-Metro --pre /home/xsh/XSH/save_file/NUPT-Metro-400-DensityAware/checkpoint.pth
```
For ShanghaiTech, you can train by a GPU with 8G memory. For other datasets, please utilize a single GPU with 24G memory or multiple GPU for training. 

**Supplementary Note:**

Data preprocessing and density map generation follow the implementation of Liang et al. For details, please refer to https://github.com/dk-liang/FIDTM




