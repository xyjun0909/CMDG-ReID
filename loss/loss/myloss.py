from doctest import FAIL_FAST
from importlib.resources import path

from numpy import tensordot
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# v4 DKL+tmp+filter+txtChange(k=10) + 原特征 + 变化权重 
class Pedal(nn.Module):
    def __init__(self, scale=10, k=10, temp_init=0.5, temp_min=0.01, 
                decay_rate=0.95, sim_threshold=0.7,epoch = 1):
        super(Pedal, self).__init__()
        self.scale = scale
        self.k = k
        self.temp = temp_init       # 初始温度
        self.temp_min = temp_min    # 最小温度
        self.decay_rate = decay_rate# 温度衰减率
        self.sim_threshold = sim_threshold  # 相似度过滤阈值
        self.epoch = epoch
    def adjust_temperature(self):
        # 每个epoch后调用，指数衰减温度
        self.temp = max(self.temp * self.decay_rate, self.temp_min)

    def epochAdd(self):
        # 每个epoch后调用，增加epoch数
        self.epoch += 1

    def align_distribution(self,img_feats, text_feats):
        num_parts = img_feats.size(0)
        text_feats = text_feats.permute(1, 0, 2)  # [3,64,768]
        align_loss = 0
        aligned_img_list = []
        
        for p in range(num_parts):
            img_part = F.normalize(img_feats[p], p=2, dim=1)  # [64,768]
            txt_part = F.normalize(text_feats[p].detach(), p=2, dim=1)
            
            # ===== 动态温度控制 =====
            img_sim = torch.mm(img_part, img_part.t()) / self.temp  # [64,64]
            txt_sim = torch.mm(txt_part, txt_part.t()) / self.temp
            
            # ===== 相似度过滤 =====
            mask = (img_sim > self.sim_threshold) & (txt_sim > self.sim_threshold)
            # 过滤无效位置（确保矩阵维度一致）
            valid_mask = mask.any(dim=1)  # [64]
            img_sim_filtered = img_sim[valid_mask][:, valid_mask]  # [M, M]
            txt_sim_filtered = txt_sim[valid_mask][:, valid_mask]
            
            # ===== 双向KL散度 =====
            if img_sim_filtered.size(0) == 0:  # 避免空矩阵
                continue
                
            img_log_prob = F.log_softmax(img_sim_filtered, dim=-1)
            txt_log_prob = F.log_softmax(txt_sim_filtered, dim=-1)
            txt_prob = F.softmax(txt_sim_filtered.detach(), dim=-1)
            img_prob = F.softmax(img_sim_filtered.detach(), dim=-1)
            
            loss_kl = 0.5 * (
                F.kl_div(img_log_prob, txt_prob, reduction='batchmean') +
                F.kl_div(txt_log_prob, img_prob, reduction='batchmean')
            )
            align_loss += loss_kl
        return align_loss
    
    def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):
        # feature 3,64,768
        # text_feature 64,3,768
        # feature = feature.permute(1, 0, 2) # 64,3,768
        # 相似度分布对齐
        align_loss = self.align_distribution(feature,text_feature)
        # feature = aligned_img_feats
        all_posvid = []
        loss = 0   

        # 动态权重：随训练进度降低KL权重
        current_epoch = self.epoch  # 获取当前epoch
        kl_weight = max(0.5 * (1 - current_epoch / 60), 0.1)  # 从0.5线性衰减到0.1 

        # 1. 跨域检索获取正样本索引 [B, 5]
        cross_indices = PatchMemory.retrieve(text_feature, domains, topk=self.k)
        cross_indices = cross_indices.to(torch.int64)
        
        # 记录正样本vid [B, 5]
        pos_vid = torch.tensor(PatchMemory.vid, device=feature.device)[cross_indices]
        all_posvid.append(pos_vid)

        for p in range(feature.size(0)):
            part_feat = feature[p]  # [B, D]
            part_centers = centers[p]  # [M, D]
            
            # 2. 获取正样本特征 [B, 5, D]
            pos_features = part_centers[cross_indices]  # 使用跨域检索结果
            
            # 3. 计算正样本距离 [B, 5]
            pos_dist = torch.cdist(part_feat.unsqueeze(1), pos_features).squeeze(1)
            
            # 4. 负样本筛选（排除正样本和自身）
            # 生成负样本掩码 [B, M]
            neg_mask = torch.ones_like(part_centers[:, 0], dtype=torch.bool)  # [M]
            neg_mask[cross_indices.flatten()] = False  # 排除正样本
            neg_mask[position] = False  # 排除自身
            
            # 5. 获取有效负样本距离 [B, M - 5 - 1]
            neg_dist = torch.cdist(part_feat, part_centers[neg_mask])  # [B, N_neg]
            
            # 6. 对比损失计算
            x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # 分子项 [B]
            y = (-self.scale * neg_dist).exp().sum(dim=1).log()  # 分母项 [B]
            # 计算基本损失
            l = (-x + y).sum().div(feature.size(1))
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
            loss += l
        contrastive_loss = contrastive_loss.div(feature.size(0))
        
        # 返回两部分损失和正样本索引
        return {
            'total_loss': contrastive_loss + kl_weight * align_loss,
            'part_contrastive_loss': contrastive_loss,
            'align_loss': align_loss,
            'all_posvid': all_posvid
        }