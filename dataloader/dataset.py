import os
import sys
import torch
from PIL import Image
import pandas as pd
import numpy as np
import paddle
from torch.utils.data import DataLoader, Dataset
from dataloader.transform import OctTransform, SloTransform
from dataloader.imp import runpredict
from PIL import Image
import torch
from torchvision import transforms


transform_resize_crop = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),  # Convert to tensor
        ])

class AierDataset(Dataset):
    def __init__(self, cfgs, mode="train"):
        super(AierDataset, self).__init__()
        self.cfgs = cfgs
        self.root_dir = self.cfgs['base_cfg']['root']
        self.mode = mode
        self.item_list = self.__parseDataset__()
        self.octTransform = OctTransform(cfgs, mode = mode)
        self.sloTransform = SloTransform(cfgs, mode = mode)

    def __getitem__(self, index):
        row = self.item_list.iloc[index]
        missModalTag = [1, 1, 1] # patientMsg、OCT、SLO
        if(pd.isna(row.OCT图像)):
            missModalTag[1] = 0
        if (pd.isna(row.SLO图像)):
            missModalTag[2] = 0
        # 用于生成分割后与分割前的结合octImage，并保存在OCTL中
        # img_path = os.path.join(self.root_dir, row.图像基础路径, row.OCT图像)
        # runpredict(img_path)

        OctImage=self.loadImage((os.path.join(self.root_dir, row.图像基础路径)), "OCTL.png")
        OctImage=self.octTransform(OctImage)
        SloImage = self.loadImage((os.path.join(self.root_dir, row.图像基础路径)), row.SLO图像)
        SloImage = self.sloTransform(SloImage, missModalTag[2])
        label = torch.tensor(row.术后LogMAR)
        patientMessage = torch.load(os.path.join(os.path.join(self.root_dir, row.图像基础路径), "patient.pt"),weights_only=True)
        diagOct = torch.load(os.path.join(os.path.join(self.root_dir, row.图像基础路径), "diagOct.pt"),weights_only=True)
        diagSlo = torch.load(os.path.join(os.path.join(self.root_dir, row.图像基础路径), "diagSlo.pt"),weights_only=True)
        return OctImage, SloImage, label, patientMessage, diagOct, diagSlo, missModalTag
    def __len__(self):
        return len(self.item_list)
    def __parseDataset__(self):
        self.txt = self.cfgs['base_cfg']['txt']
        dataList = pd.read_csv(os.path.join(self.root_dir, self.txt), encoding = "utf-8")
        seed = self.cfgs['base_cfg']['seed']
        dataList = dataList.sample(frac = 1, random_state = seed, replace = False)
        if self.mode == "train":
            data = dataList[0: int(len(dataList) * 0.8)]
        elif self.mode == "val":
            data = dataList[int(len(dataList) * 0.8): int(len(dataList) * 0.9)]
        else:
            data = dataList[int(len(dataList) * 0.9): int(len(dataList))]
        return data

    def loadImage(self, imageBasePath, imageName):
        if(pd.isna(imageName)):
            img = np.zeros((512, 512, 3), dtype = np.uint8)
            img = Image.fromarray(img)
        else:
            img = Image.open(os.path.join(imageBasePath, imageName)).convert("RGB")
        return img

def dataloader(dataset, cfgs):
    return DataLoader(dataset = dataset,
                    batch_size = cfgs['train_cfg']['Batch_Size'],
                    shuffle = True,
                    pin_memory = True,
                    num_workers = 8)


