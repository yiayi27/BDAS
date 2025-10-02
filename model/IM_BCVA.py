
import torch
from einops import repeat, rearrange
from torch import nn
from model.OctModel import OctNet
from model.SloModel import SloNet
import torch.nn as nn
import torch.nn.functional as F


# 用于随机遮蔽的函数（适用于所有模态）
def random_mask_slo(x, mask_prob=0.2, full_mask_prob=0.3):
    device = x.device
    if torch.rand(1).item() < full_mask_prob:
        mask = torch.zeros_like(x)  # 完全遮蔽
    else:
        mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > mask_prob).float()  # 随机遮蔽
    return x * mask  # 将被遮蔽的部分置为0

def random_mask(x,mask_prob=0.2):
    device=x.device
    mask = (torch.rand(x.shape[0], x.shape[1], 1, device=device) > mask_prob).float()
    return x*mask

# 定义多粒度跨模态交互模块（MCIM）
class MCIM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MCIM, self).__init__()

        # 粗粒度：全局注意力层
        self.coarse_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)

        # 细粒度：局部注意力层
        self.fine_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # 输入的维度调整
        x_orig = x
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_dim]

        # 粗粒度交互：全局注意力
        coarse_output, _ = self.coarse_attention(x, x, x)
        coarse_output = self.norm1(coarse_output + x)

        # 细粒度交互：局部注意力
        fine_output, _ = self.fine_attention(coarse_output, coarse_output, coarse_output)
        fine_output = self.norm2(fine_output + coarse_output)

        # 维度调整回原始形状
        fine_output = fine_output.permute(1, 0, 2)  # [batch_size, seq_len, input_dim]

        # 前馈网络
        output = self.ffn(fine_output)
        output = self.norm3(output + fine_output)

        return output


class DFIM(nn.Module):
    def __init__(self, input_dim):
        super(DFIM, self).__init__()
        # 自增强
        self.fc = nn.Linear(input_dim, 1)

        # 选择性过滤
        self.filter = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, feature1, feature2):
        # 帧级自增强 - 使用全连接层计算权重
        a_scores = self.fc(feature1)  # [batch_size, 1]
        b_scores = self.fc(feature2)  # [batch_size, 1]

        a_weights = F.softmax(a_scores, dim=0)
        b_weights = F.softmax(b_scores, dim=0)

        h_a_enhanced = feature1 * a_weights
        h_b_enhanced = feature2 * b_weights

        # 选择性过滤
        concat = torch.cat([h_a_enhanced, h_b_enhanced], dim=1)  # [batch_size, dim*2]
        mu = self.filter(concat)  # [batch_size, dim]

        h_e = mu * h_a_enhanced + (1 - mu) * h_b_enhanced  # [batch_size, dim]
        h_e = h_e.unsqueeze(1)
        return h_e  # 直接返回融合后的特征 [batch_size, dim]



class UMDF_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(UMDF_Model, self).__init__()
        self.mcim = MCIM(input_dim, hidden_dim)
        self.dfim = DFIM(input_dim)

        # 预测器 - 注意输入维度是 input_dim，不需要挤压
        self.pred = nn.Linear(input_dim, 1)

    def forward(self, octEmbeded, sloEmbeded, textEmbeded):
        sloEmbeded_masked1 = random_mask_slo(sloEmbeded)
        sloEmbeded_masked2 = random_mask_slo(sloEmbeded)
        slo_features=self.mcim(sloEmbeded)
        oct_features = self.mcim(octEmbeded)
        slo_features1 = self.mcim(sloEmbeded_masked1)
        text_features = self.mcim(textEmbeded)
        feature=torch.mean(oct_features,dim=1)+torch.mean(slo_features,dim=1)+torch.mean(text_features,dim=1)
        feature1 = torch.mean(oct_features, dim=1) + torch.mean(slo_features1, dim=1) + torch.mean(text_features, dim=1)
        slo_features2 = self.mcim(sloEmbeded_masked2)
        feature2 = torch.mean(oct_features, dim=1) + torch.mean(slo_features2, dim=1) + torch.mean(text_features, dim=1)
        predBCVA1 = self.pred(feature1)
        predBCVA2 = self.pred(feature2)
        integrated_feature = self.dfim(feature1, feature2)  # [batch_size, 1, dim]
        predBCVA = self.pred(integrated_feature[:, 0, :])  # 取第一个时间步的特征
        predBCVA1=self.pred(feature)
        feature1=feature
        feature2=feature
        predBCVA2=predBCVA1
        return feature1, feature2, predBCVA1, predBCVA2, integrated_feature[:, 0, :]


class IncompleteBCVA(nn.Module):
    def __init__(self, cfgs):
        super(IncompleteBCVA, self).__init__()
        dim = cfgs['model_cfg']['incomplete_fusion']['dim']
        self.model = UMDF_Model(input_dim=768, hidden_dim=512)
        classes = cfgs['model_cfg']['BCVA_Num_Classes']
        self.cfgs = cfgs
        self.octEncoder = OctNet(cfgs)
        self.sloEncoder = SloNet(cfgs)

        # 预测器 - 确保输入维度正确
        self.pred = nn.Sequential(
            nn.LayerNorm(768),  # 使用正确的维度
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(768, classes),
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self,
                OctImage,  # [batchSize, channel, h, w]
                patientMessage,  # [batchSize, seqLength, dim]
                SloImage,  # [batchSize, channel, h, w]
                MissingLabel,  # [batchSize, 3]
                diagOct,
                diagSlo
                ):
        octEmbed, predOct = self.octEncoder(OctImage)
        sloEmbed, predSlo = self.sloEncoder(SloImage)
        textEmbed = patientMessage

        # 使用 self.model 来调用 UMDF_Model
        feature1, feature2, predBCVA1, predBCVA2, integrated_feature = self.model(octEmbed, sloEmbed, textEmbed)

        # 确保维度正确后再使用预测器
        # 不需要在这里挤压维度，因为 self.model 已经返回了正确形状的 feature1, feature2, integrated_feature
        predBCVA1_final = self.pred(feature1)
        predBCVA2_final = self.pred(feature2)
        predBCVA_final = self.pred(integrated_feature)

        # 确保输出维度正确
        return (feature1, feature2,
                predBCVA1_final.squeeze(-1),
                predBCVA2_final.squeeze(-1),
                predBCVA_final.squeeze(-1),
                predOct, predSlo, octEmbed, sloEmbed)





