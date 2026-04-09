import os

config_path = './Networks/HR_Net/seg_hrnet_w48.yaml'
if os.path.exists(config_path):
    print("配置文件路径正确")
else:
    print("配置文件路径错误")