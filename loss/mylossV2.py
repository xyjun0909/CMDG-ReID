import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class Pedal(nn.Module):
    """多模态对比损失类（优化版本）：支持高相似样本筛选的MMD"""

    def __init__(
        self,
        scale=10,  # 局部对比损失的缩放因子
        k=10,  # 检索的相似样本数量
        kernel_bandwidth=1.0,  # MMD核带宽（控制核函数平滑度，核心参数）
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
        self.batch_count = 0  # 批次计数器
        self.total_batches = 0  # 总批次计数器
        self.mmd_loss_history = []  # 记录MMD损失，便于调试

    def _rbf_kernel(self, x, y):
        """计算RBF核矩阵（MMD的核心）"""
        dist_matrix = torch.cdist(x, y, p=2) ** 2  # [B1, B2]
        return torch.exp(-dist_matrix / (2 * self.kernel_bandwidth**2))

    def align_global_distribution(
        self, 
        global_img_feat, 
        global_text_feat,  # 现在是高相似样本的文本特征：[B*K, 512]
        domains=None, 
        captions=None,
        batch_topk_indices=None  # 新增：当前batch的高相似索引，用于验证
    ):
        """
        优化后的全局分布对齐：使用高相似文本样本计算MMD
        原理：用 [B,512] 图像特征 对比 [B*K,512] 高相似文本特征，MMD更聚焦关键分布
        """
        batch_size_img = global_img_feat.size(0)  # 图像样本数：B
        batch_size_txt = global_text_feat.size(0)  # 文本样本数：B*K
        self.batch_count += 1
        self.total_batches += 1

        # 1. 特征归一化（确保在单位球上计算距离，关键步骤）
        img_feat = F.normalize(global_img_feat, p=2, dim=1)  # [B, D]
        txt_feat = F.normalize(global_text_feat.detach(), p=2, dim=1)  # [B*K, D]

        # 2. 计算MMD的三个核心核矩阵（适配高相似文本样本的维度）
        k_xx = self._rbf_kernel(img_feat, img_feat)  # [B, B]：图像自身相似度
        k_yy = self._rbf_kernel(txt_feat, txt_feat)  # [B*K, B*K]：高相似文本自身相似度
        k_xy = self._rbf_kernel(img_feat, txt_feat)  # [B, B*K]：图像-高相似文本相似度

        # 3. 移除对角线元素（排除自身与自身的相似度，避免高估分布一致性）
        # 图像核矩阵：[B, B] → 移除对角线后平均
        mask_img = ~torch.eye(batch_size_img, device=img_feat.device, dtype=torch.bool)
        k_xx = k_xx[mask_img].view(batch_size_img, batch_size_img - 1).mean()
        # 文本核矩阵：[B*K, B*K] → 移除对角线后平均（高相似样本无自身重叠，可直接平均）
        mask_txt = ~torch.eye(batch_size_txt, device=img_feat.device, dtype=torch.bool)
        k_yy = k_yy[mask_txt].view(batch_size_txt, batch_size_txt - 1).mean()
        # 交叉核矩阵：直接平均（无自身重叠问题）
        k_xy = k_xy.mean()

        # 4. 计算MMD损失（非负，值越小分布越对齐）
        mmd_loss = k_xx + k_yy - 2 * k_xy

        # 5. 记录损失（用于调试和参数调整）
        self.mmd_loss_history.append(mmd_loss.item())

        # 6. 日志：每100批次打印一次MMD损失（新增筛选后样本数）
        if self.total_batches % 100 == 0:
            print(
                f"批次 {self.total_batches} | MMD损失: {mmd_loss.item():.4f} | "
                f"核带宽: {self.kernel_bandwidth} | "
                f"图像样本数: {batch_size_img} | 高相似文本样本数: {batch_size_txt}"
            )

        return mmd_loss

    def forward(
        self,
        global_img_feat,
        global_text_feat,  # 现在是高相似样本的文本特征：[B*K, 512]
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
        batch_topk_indices=None  # 新增：当前batch的高相似索引（可选，用于后续扩展）
    ):
        """前向传播：整合全局MMD损失（带筛选）和局部对比损失"""
        # 1. 全局分布对齐损失（MMD）：使用高相似文本样本
        align_loss = self.align_global_distribution(
            global_img_feat, 
            global_text_feat, 
            domains, 
            captions,
            batch_topk_indices=batch_topk_indices
        )

        # 2. 局部特征对比损失（保持原有逻辑，确保局部特征判别性）
        all_posvid = []
        local_loss = 0.0
        mmd_weight = 1.0  # MMD损失权重（可根据效果调整，建议0.8-1.2）

        # 计算每个局部区域的对比损失（原有逻辑不变）
        for p in range(local_img_feats.size(0)):
            part_feat = local_img_feats[p, :, :]
            part_centers = centers[p, :, :]
            m, n = part_feat.size(0), part_centers.size(0)
            dist_map = (
                part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n)
                + part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            )
            dist_map.addmm_(1, -2, part_feat, part_centers.t())

            trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)

            neg, index = (
                dist_map[trick != position.unsqueeze(dim=1).expand_as(dist_map)]
                .view(dist_map.size(0), -1)
                .sort(dim=1)
            )

            pos_camid = torch.tensor(PatchMemory.camid).cuda()
            pos_camid = pos_camid[(index[:, : self.k])]
            flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)

            pos_vid = torch.tensor(PatchMemory.vid).cuda()
            pos_vid = pos_vid[(index[:, : self.k])]
            all_posvid.append(pos_vid)

            x = ((-1 * self.scale * neg[:, : self.k]).exp().sum(dim=1)).log()
            y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
            l = (-x + y).sum().div(local_img_feats.size(1))
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.0), l)
            local_loss += l

        # 平均所有局部区域的损失（原有逻辑不变）
        local_loss = local_loss.div(local_img_feats.size(0))

        # 总损失 = 局部对比损失 + MMD损失（带权重）（原有逻辑不变）
        total_loss = local_loss + mmd_weight * align_loss

        return {
            "total_loss": total_loss,
            "part_contrastive_loss": local_loss,
            "align_loss": align_loss,
            "all_posvid": all_posvid,
        }