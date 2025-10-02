from einops import rearrange, repeat
from torch import nn
import torch
import torch.nn.functional as F
class OctNet(nn.Module):
    def __init__(self, cfgs):
        super(OctNet, self).__init__()
        dim = cfgs['model_cfg']['incomplete_fusion']['dim']
        classes = cfgs['model_cfg']['BCVA_Num_Classes']
        self.omega = nn.Parameter(torch.ones(1, requires_grad = True))
        self.backbone = nn.Sequential(nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = 3),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace = True),
                                      nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                                      )
        self.transform = nn.Sequential(nn.Conv2d(3, 64, kernel_size = (7, 7), stride = (2, 2), padding = 3),
                                       nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                                       )

        self.block1 = Block(64, 128)
        self.block2 = Block(128, 256)
        self.block3 = Block(256, dim, isBottleneck = False)
        self.pred = nn.Sequential(nn.LayerNorm(dim),
                                  # nn.Tanh(),
                                  nn.ReLU(),
                                  nn.Dropout(p = 0.5),
                                  nn.Linear(dim, classes),
                                  )
        self.conv = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.backbone(x)  # [b, 64, 56, 56]
        x = self.block1(x) # [b, 128, 28, 28]
        x = self.block2(x) # [b, 256, 14, 14]
        x = self.block3(x) # [b, 384, 7, 7]
        embed = x.flatten(start_dim = 2).permute(0, 2, 1) # [b, 49, 384]
        predOct = self.pred(F.adaptive_avg_pool2d(x,(1,1)).squeeze())
        return embed, predOct.squeeze()


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, isBottleneck = True):
        super(Block, self).__init__()
        self.isBottleneck = isBottleneck
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = (3, 3), stride = (2, 2), padding = (1, 1)),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU(),
                                   )
        self.downSample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = (1, 1), stride = (2, 2)),
                                        nn.BatchNorm2d(out_channel),
                                        nn.ReLU(),
                                        )
        if isBottleneck:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(out_channel, out_channel // 4, kernel_size = (1, 1), stride = (1, 1)),
                nn.BatchNorm2d(out_channel // 4),
                nn.ReLU(),
                nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
                nn.BatchNorm2d(out_channel // 4),
                nn.ReLU(),
                nn.Conv2d(out_channel // 4, out_channel, kernel_size = (1, 1), stride = (1, 1)),
                nn.BatchNorm2d(out_channel),
                )
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        res = self.downSample(identity)
        x = x + res
        if self.isBottleneck:
            x = self.bottleneck(x)
        return x

