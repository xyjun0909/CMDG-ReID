import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class Pedal(nn.Module):
    
    def __init__(
        self,
        scale=10,
        k=10,
        kernel_bandwidth=1.0,
        epoch=1,
        log_dir=None,
        log_freq=10,
        use_optimized_scheme=False  
    ):
        super(Pedal, self).__init__()
        self.scale = scale
        self.k = k
        self.kernel_bandwidth = kernel_bandwidth
        self.epoch = epoch
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.batch_count = 0
        self.total_batches = 0
        self.mmd_loss_history = []
        self.use_optimized = use_optimized_scheme  

    def _rbf_kernel(self, x, y):
        dist_matrix = torch.cdist(x, y, p=2) ** 2  # [B1, B2]
        return torch.exp(-dist_matrix / (2 * self.kernel_bandwidth**2))

    def align_global_distribution(
        self, 
        global_img_feat, 
        global_text_feat,  
        domains=None, 
        captions=None,
        batch_topk_indices=None  
    ):
        batch_size_img = global_img_feat.size(0)
        batch_size_txt = global_text_feat.size(0)
        self.batch_count += 1
        self.total_batches += 1

        img_feat = F.normalize(global_img_feat, p=2, dim=1)
        txt_feat = F.normalize(global_text_feat.detach(), p=2, dim=1)

        if not self.use_optimized:
            assert batch_size_img == batch_size_txt, "原始方案要求文本和图像样本数相同"
            
            k_xx = self._rbf_kernel(img_feat, img_feat)
            k_yy = self._rbf_kernel(txt_feat, txt_feat)
            k_xy = self._rbf_kernel(img_feat, txt_feat)
            
            mask = ~torch.eye(batch_size_img, device=img_feat.device, dtype=torch.bool)
            k_xx = k_xx[mask].view(batch_size_img, batch_size_img - 1).mean()
            k_yy = k_yy[mask].view(batch_size_img, batch_size_img - 1).mean()
            k_xy = k_xy.mean()

        else:
            assert batch_size_txt == batch_size_img * batch_topk_indices.size(1), \
                f"优化方案文本样本数应为 {batch_size_img}×{batch_topk_indices.size(1)}，实际为 {batch_size_txt}"
            
            k_xx = self._rbf_kernel(img_feat, img_feat)  # [B, B]
            k_yy = self._rbf_kernel(txt_feat, txt_feat)  # [B*K, B*K]
            k_xy = self._rbf_kernel(img_feat, txt_feat)  # [B, B*K]
            
            mask_img = ~torch.eye(batch_size_img, device=img_feat.device, dtype=torch.bool)
            k_xx = k_xx[mask_img].view(batch_size_img, batch_size_img - 1).mean()
            
            mask_txt = ~torch.eye(batch_size_txt, device=img_feat.device, dtype=torch.bool)
            k_yy = k_yy[mask_txt].view(batch_size_txt, batch_size_txt - 1).mean()
            
            k_xy = k_xy.mean()

        mmd_loss = k_xx + k_yy - 2 * k_xy
        self.mmd_loss_history.append(mmd_loss.item())

        if self.total_batches % 100 == 0:
            if self.use_optimized:
                print(
                    f"批次 {self.total_batches} | 优化方案MMD损失: {mmd_loss.item():.4f} | "
                    f"核带宽: {self.kernel_bandwidth} | 图像样本数: {batch_size_img} | "
                    f"高相似文本样本数: {batch_size_txt}"
                )
            else:
                print(
                    f"批次 {self.total_batches} | 原始方案MMD损失: {mmd_loss.item():.4f} | "
                    f"核带宽: {self.kernel_bandwidth} | 样本数: {batch_size_img}"
                )

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
        batch_topk_indices=None 
    ):
        align_loss = self.align_global_distribution(
            global_img_feat, 
            global_text_feat, 
            domains, 
            captions,
            batch_topk_indices=batch_topk_indices
        )

        all_posvid = []
        local_loss = 0.0
        mmd_weight = 1.0

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

        local_loss = local_loss.div(local_img_feats.size(0))
        total_loss = local_loss + mmd_weight * align_loss

        return {
            "total_loss": total_loss,
            "part_contrastive_loss": local_loss,
            "align_loss": align_loss,
            "all_posvid": all_posvid,
        }