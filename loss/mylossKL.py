import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class Pedal(nn.Module):
    """多模态对比损失函数类（无阈值版本），使用KL散度对齐全局特征分布"""

    def __init__(
        self,
        scale=10,
        k=10,
        temp_init=0.5,  # 提高初始温度，避免相似度极化
        temp_min=0.2,   # 提高最小温度，保持分布平滑
        decay_rate=0.98,  # 放缓温度衰减
        epoch=1,
        log_dir=None,
        log_freq=10,
    ):
        super(Pedal, self).__init__()
        self.scale = scale  # 局部对比损失的缩放因子
        self.k = k          # 检索的相似样本数量
        self.temp = temp_init  # 温度参数（控制相似度矩阵平滑度）
        self.temp_min = temp_min
        self.decay_rate = decay_rate
        self.epoch = epoch
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.batch_count = 0  # 批次计数器
        self.total_batches = 0  # 总批次计数器

    def adjust_temperature(self):
        """每个epoch调整温度参数（缓慢衰减）"""
        self.temp = max(self.temp * self.decay_rate, self.temp_min)

    def epochAdd(self):
      
        self.epoch += 1
        self.batch_count = 0  # 每个epoch重置批次计数器

    def align_global_distribution(self, global_img_feat, global_text_feat, domains=None, captions=None):
        """
        对齐全局图像特征和全局文本特征的分布（无阈值过滤）
        """
        batch_size = global_img_feat.size(0)
        self.batch_count += 1
        self.total_batches += 1
        
        # 特征归一化（确保在单位球上计算相似度）
        img_feat = F.normalize(global_img_feat, p=2, dim=1)  # [B, D]
        txt_feat = F.normalize(global_text_feat.detach(), p=2, dim=1)  # [B, D]

        # 计算全局相似度矩阵（无阈值过滤，保留所有样本对）
        img_sim = torch.mm(img_feat, img_feat.t()) / self.temp  # [B, B]
        txt_sim = torch.mm(txt_feat, txt_feat.t()) / self.temp  # [B, B]

        # 移除对角线（自身与自身的相似度，避免影响分布计算）
        mask = ~torch.eye(batch_size, device=img_feat.device, dtype=torch.bool)
        img_sim = img_sim[mask].view(batch_size, batch_size - 1)  # [B, B-1]
        txt_sim = txt_sim[mask].view(batch_size, batch_size - 1)  # [B, B-1]

        # 计算双向KL散度（使用所有样本对）
        img_log_prob = F.log_softmax(img_sim, dim=-1)
        txt_log_prob = F.log_softmax(txt_sim, dim=-1)
        txt_prob = F.softmax(txt_sim.detach(), dim=-1)  # 文本分布作为目标
        img_prob = F.softmax(img_sim.detach(), dim=-1)  # 图像分布作为目标

        # 双向KL散度（平均两个方向的差异）
        loss_kl = 0.5 * (
            F.kl_div(img_log_prob, txt_prob, reduction="batchmean") +
            F.kl_div(txt_log_prob, img_prob, reduction="batchmean")
        )
        
        # 日志：每100批次打印一次KL损失和有效样本数
        if self.total_batches % 100 == 0:
            print(f"KL散度损失: {loss_kl.item():.4f}, 温度: {self.temp:.3f}, 批次样本数: {batch_size}")
        
        return loss_kl

    def forward(
        self,
        global_img_feat,
        global_text_feat,
        local_img_feats,
        local_text_feats,
        centers,
        text_centers,
        position,
        PatchMemory=None,
        vid=None,
        camid=None,
        domains=None,
        captions=None,
    ):
        """计算多模态损失（全局KL散度 + 局部对比损失）"""
        # 1. 全局特征分布对齐损失（无阈值KL散度）
        align_loss = self.align_global_distribution(
            global_img_feat, global_text_feat, domains, captions
        )

        # 2. 局部特征对比损失（保持原有逻辑）
        all_posvid = []
        local_loss = 0.0
        kl_weight = 0.5  # 全局损失权重

        # 从内存银行检索跨域相似样本
        cross_indices = PatchMemory.retrieve_all_domains(local_text_feats, topk=self.k)
        cross_indices = cross_indices.to(torch.int64)

        # 获取正样本ID（根据内存库中的vid）
        pos_vid = torch.tensor(PatchMemory.vid, device=global_img_feat.device)[cross_indices]
        all_posvid.append(pos_vid)

        # 计算每个局部区域的对比损失
        for p in range(local_img_feats.size(0)):
            part_feat = local_img_feats[p]  # [B, D]
            part_centers = centers[p]       # [N, D]（内存库中的局部中心）
            pos_features = part_centers[cross_indices]  # [B, k, D]
            pos_dist = torch.cdist(part_feat.unsqueeze(1), pos_features).squeeze(1)  # [B, k]

            # 构建负样本掩码（排除正样本和自身）
            neg_mask = torch.ones_like(part_centers[:, 0], dtype=torch.bool)
            neg_mask[cross_indices.flatten()] = False  # 排除正样本
            neg_mask[position] = False  # 排除自身
            neg_dist = torch.cdist(part_feat, part_centers[neg_mask])  # [B, M]（M为负样本数）

            # 局部对比损失计算（InfoNCE变体）
            x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # 正样本贡献
            y = (-self.scale * neg_dist).exp().sum(dim=1).log()  # 负样本贡献
            l = (-x + y).sum().div(part_feat.size(0))  # 平均到每个样本
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.0), l)  # 处理NaN
            local_loss += l

        # 平均所有局部区域的损失
        local_loss = local_loss.div(local_img_feats.size(0))
        
        return {
            'total_loss': local_loss + kl_weight * align_loss,
            'part_contrastive_loss': local_loss,
            'align_loss': align_loss,
            'all_posvid': all_posvid
        }
