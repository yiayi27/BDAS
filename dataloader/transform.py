import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

class OctTransform(object):
    def __init__(self, cfgs, img_size = 256, crop_size = 224, mode="train"):
        self.image_size = (crop_size, crop_size)
        self.mode = mode
        self.cfgs = cfgs
        mean = cfgs['train_cfg']['oct']['IMG_MEAN']
        std = cfgs['train_cfg']['oct']['IMG_STD']
        self.transform1 = transforms.Compose([
                                              transforms.Resize((img_size, img_size)),
                                              transforms.CenterCrop(crop_size),
                                              ])
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean = mean, std = std)])
        self.transform3 = transforms.ToTensor()

    def __call__(self, x):
        x = self.transform1(x)
        x = self.transform2(x)
        return x.type(torch.FloatTensor)

class SloTransform(object):
    def __init__(self, cfgs, img_size = 224, mode = "train"):
        mean = cfgs['train_cfg']['slo']['IMG_MEAN']
        std = cfgs['train_cfg']['slo']['IMG_STD']
        self.divsor = 255.0
        self.mode = mode
        self.transform1 = transforms.Resize((img_size,img_size))
        self.transform2 = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean = mean, std = std)])
    def __call__(self, x, m_label):
        x = self.transform1(x)
        x = np.array(x) / self.divsor
        x = self.transform2(x)
        return x.type(torch.FloatTensor)

