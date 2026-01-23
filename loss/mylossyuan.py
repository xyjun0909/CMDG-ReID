import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class Pedal(nn.Module):
    """多模态对比损失类（完整版本）：使用MMD对齐全局分布 + 局部对比损失"""

    def __init__(
        self,
        scale=10,               # 局部对比损失的缩放因子
        k=10,                   # 检索的相似样本数量
        kernel_bandwidth=1.0,   # MMD核带宽（控制核函数平滑度，核心参数）
        epoch=1,
        log_dir=None,
        log_freq=10,
    ):
        super(Pedal, self).__init__()
        self.scale = scale
        self.k = k
        self.kernel_bandwidth = kernel_bandwidth  # MMD核心参数，建议0.5-2.0
        self.epoch = epoch
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.batch_count = 0          # 批次计数器
        self.total_batches = 0        # 总批次计数器
        self.mmd_loss_history = []    # 记录MMD损失，便于调试


    def _rbf_kernel(self, x, y):
        """
        计算RBF核矩阵（MMD的核心）
        x: 特征矩阵 [B1, D]
        y: 特征矩阵 [B2, D]
        返回：核矩阵 [B1, B2]
        """
        # 计算欧氏距离平方
        dist_matrix = torch.cdist(x, y, p=2) ** 2  # [B1, B2]
        # RBF核：exp(-距离² / (2*带宽²))
        return torch.exp(-dist_matrix / (2 * self.kernel_bandwidth **2))

    def align_global_distribution(self, global_img_feat, global_text_feat, domains=None, captions=None):
        """
        核心函数：使用MMD对齐全局图像和文本特征分布
        原理：MMD值越小，两个分布越相似
        """
        batch_size = global_img_feat.size(0)
        self.batch_count += 1
        self.total_batches += 1

        # 1. 特征归一化（确保在单位球上计算距离，关键步骤）
        img_feat = F.normalize(global_img_feat, p=2, dim=1)  # [B, D]
        txt_feat = F.normalize(global_text_feat.detach(), p=2, dim=1)  # [B, D]

        # 2. 计算MMD的三个核心核矩阵
        # 图像特征自身的核矩阵 [B, B]
        k_xx = self._rbf_kernel(img_feat, img_feat)
        # 文本特征自身的核矩阵 [B, B]
        k_yy = self._rbf_kernel(txt_feat, txt_feat)
        # 图像与文本的交叉核矩阵 [B, B]
        k_xy = self._rbf_kernel(img_feat, txt_feat)

        # 3. 移除对角线元素（排除自身与自身的相似度，避免高估分布一致性）
        mask = ~torch.eye(batch_size, device=img_feat.device, dtype=torch.bool)
        k_xx = k_xx[mask].view(batch_size, batch_size - 1).mean()  # 平均图像核值
        k_yy = k_yy[mask].view(batch_size, batch_size - 1).mean()  # 平均文本核值
        k_xy = k_xy.mean()  # 平均交叉核值

        # 4. 计算MMD损失（非负，值越小分布越对齐）
        mmd_loss = k_xx + k_yy - 2 * k_xy

        # 5. 记录损失（用于调试和参数调整）
        self.mmd_loss_history.append(mmd_loss.item())

        # 6. 日志：每100批次打印一次MMD损失
        if self.total_batches % 100 == 0:
            print(f"批次 {self.total_batches} | MMD损失: {mmd_loss.item():.4f} | 核带宽: {self.kernel_bandwidth}")

        return mmd_loss

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
        """前向传播：整合全局MMD损失和局部对比损失"""
        # 1. 全局分布对齐损失（MMD）
        align_loss = self.align_global_distribution(
            global_img_feat, global_text_feat, domains, captions
        )

        # 2. 局部特征对比损失（保持原有逻辑，确保局部特征判别性）
        all_posvid = []
        local_loss = 0.0
        mmd_weight = 1.0  # MMD损失权重（可根据效果调整，建议0.8-1.2）

        # 从内存银行检索跨域相似样本
        cross_indices = PatchMemory.retrieve(local_text_feats, topk=self.k)
        cross_indices = cross_indices.to(torch.int64)

        # 获取正样本ID（用于ReID损失的软标签）
        # todo: change
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
            neg_dist = torch.cdist(part_feat, part_centers[neg_mask])  # [B, M]

            # 局部对比损失计算（InfoNCE变体）
            x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # 正样本贡献
            y = (-self.scale * neg_dist).exp().sum(dim=1).log()  # 负样本贡献
            l = (-x + y).sum().div(part_feat.size(0))  # 平均到每个样本
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.0), l)  # 处理NaN
            local_loss += l

        # 平均所有局部区域的损失
        local_loss = local_loss.div(local_img_feats.size(0))

        # 总损失 = 局部对比损失 + MMD损失（带权重）
        total_loss = local_loss + mmd_weight * align_loss

        return {
            'total_loss': total_loss,
            'part_contrastive_loss': local_loss,
            'align_loss': align_loss,  # 此处为MMD损失
            'all_posvid': all_posvid
        }
