import torch
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys


class ContrastiveLoss(nn.Module):
    def __init__(self, cfgs):
        super(ContrastiveLoss, self).__init__()
        self.temperature = cfgs['train_cfg']['incomplete']['tau']
        self.Lambda = cfgs['train_cfg']['incomplete']['lambda']
        self.isLossMask = cfgs['base_cfg']['isLossMask']

    def forward(self, image, text, missModalTag, type):
        # 增加模块内的相似性，保留跨模态任务中，同模态信息对齐
        assert type == "OCT" or type == "SLO", "Contrastive Learning type is error!!!"
        if type == "OCT":
            mask =  missModalTag[1]
        else:
            mask =  missModalTag[2]
        image, text = image.mean(1), text.mean(1)
        image = F.normalize(image, p = 2, dim = 1)
        text = F.normalize(text, p = 2, dim   = 1)
        batch_size = image.shape[0]

        similarity_matrix_xy = torch.exp(F.cosine_similarity(image.unsqueeze(1), text.unsqueeze(0), dim = 2) / self.temperature)
        similarity_matrix_xx = torch.exp(F.cosine_similarity(image.unsqueeze(0), image.unsqueeze(0), dim = 2) / self.temperature)
        similarity_matrix_yx=torch.exp(F.cosine_similarity(text.unsqueeze(1), image.unsqueeze(0), dim = 2) / self.temperature)
        similarity_matrix_yy=torch.exp(F.cosine_similarity(text.unsqueeze(0), text.unsqueeze(0), dim = 2) / self.temperature)

        sum_row_xy = torch.sum(similarity_matrix_xy, dim = 1)
        sum_row_xx=torch.sum(similarity_matrix_xx, dim=1)
        sum_col_yy=torch.sum(similarity_matrix_yy, dim=0)
        sum_col_yx=torch.sum(similarity_matrix_yx, dim=0)

        loss_it = torch.diag(- torch.log(torch.div(similarity_matrix_xy, sum_row_xy[:, None]+1.2*sum_row_xx[:,None])))
        loss_ti = torch.diag(- torch.log(torch.div(similarity_matrix_yx, sum_col_yx[:, None]+1.2*sum_col_yy[:,None])))
        if self.isLossMask:
            loss = torch.sum(mask * (self.Lambda * loss_ti + (1 - self.Lambda) * loss_it)) / batch_size
        else:
            loss = torch.sum(self.Lambda * loss_ti + (1 - self.Lambda) * loss_it) / batch_size
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, y_pred, y_true):
        mse = torch.mean(torch.square(y_pred - y_true))
        return mse


# 改进的特征一致性损失，使用余弦相似度
class FeatureConsistencyLoss(nn.Module):
    def __init__(self):
        super(FeatureConsistencyLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos_sim = F.cosine_similarity(feature1, feature2, dim=-1)
        loss = 1 - cos_sim.mean()  # 平均相似度损失
        return loss


class LogitsConsistencyLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(LogitsConsistencyLoss, self).__init__()
        self.sigma = sigma  # 可调节的平滑参数

    def forward(self, pred1, pred2):
        # 1. 基础MSE损失 - 直接度量两个预测值之间的均方误差
        mse_loss = F.mse_loss(pred1, pred2)

        # 2. 相对误差损失 - 考虑预测值的相对差异，对较大视力值的小差异更宽容
        # 避免分母为0
        eps = 1e-8
        mean_pred = (torch.abs(pred1) + torch.abs(pred2)) / 2 + eps
        relative_diff = torch.abs(pred1 - pred2) / mean_pred
        relative_loss = torch.mean(relative_diff)

        # 3. 平滑L1损失 - 在小误差处更平滑，对异常值不那么敏感
        smooth_l1_loss = F.smooth_l1_loss(pred1, pred2, beta=0.1)

        # 组合损失 - 根据任务需求可调整权重
        combined_loss = mse_loss + 0.2 * relative_loss + 0.3 * smooth_l1_loss

        return combined_loss

class TotalLoss(nn.Module):
    def __init__(self, feature_loss_weight_init=1.0, logits_loss_weight_init=1.0):
        super(TotalLoss, self).__init__()
        self.feature_loss = FeatureConsistencyLoss()
        self.logits_loss = LogitsConsistencyLoss()
        self.feature_loss_weight = 0.85
        self.logits_loss_weight = 0.15

    def forward(self, feature1, feature2, predBCVA1, predBCVA2):
        # 计算特征一致性损失
        feature_loss_value = self.feature_loss(feature1, feature2)
        # 计算logits一致性损失
        logits_loss_value = self.logits_loss(predBCVA1, predBCVA2)
        total_loss = self.feature_loss_weight*feature_loss_value +self.logits_loss_weight*logits_loss_value

        return total_loss







