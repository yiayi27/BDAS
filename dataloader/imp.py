import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import paddle
sys.path.append("/data/hanxu/BIBM_codeN/PaddleSeg")
from paddleseg.cvlibs import manager, Config
import shutil
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.core import predict
from paddleseg.transforms import Compose


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def get_test_config(cfg):
    test_config = cfg.test_config
    test_config['aug_pred'] = True  # 设置是否使用多尺度和翻转增强
    test_config['scales'] = [1.0]  # 设置放大倍数

    test_config['flip_horizontal'] = True  # 是否使用水平翻转增强
    test_config['flip_vertical'] = False  # 是否使用垂直翻转增强

    test_config['is_slide'] = False  # 是否使用滑动窗口预测
    test_config['crop_size'] = None  # 设置滑动窗口的裁剪大小
    test_config['stride'] = None  # 设置滑动窗口的步长

    test_config['custom_color'] = [0,0,0,80,80,80,160,160,160,255,255,255]  # 设置自定义颜色

    return test_config

def runpredict(image_path):
    # 设置设备
    env_info = get_sys_env()
    # 确定设备
    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'
    paddle.set_device(place)

    # 加载配置文件
    config_path = "/data/hanxu/BIBM_codeN/PaddleSeg/configs/quick_start/mine.yml"
    cfg = Config(config_path)

    # 加载模型和处理配置
    model = cfg.model
    model_path = "/data/hanxu/BIBM_codeN/PaddleSeg/output/best_model/model.pdparams"
    transforms = Compose(cfg.val_transforms)
    image_list, image_dir = get_image_list(image_path)
    save_dir = "/data/hanxu/BIBM_codeN/BIBM_code/res"
    test_config = get_test_config(cfg)

    # 进行预测
    added_image,pred_mask=predict(
        model,
        model_path=model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=save_dir,
        **test_config)


    added_image_path = os.path.join(os.path.dirname(image_path), "OCTL.png")
    mkdir(added_image_path)
    cv2.imwrite(added_image_path, added_image)

    #
    # pred_saved_path = os.path.join(
    #     os.path.dirname(image_path), "PRED.png")
    # mkdir(pred_saved_path)
    # pred_mask.save(pred_saved_path)

