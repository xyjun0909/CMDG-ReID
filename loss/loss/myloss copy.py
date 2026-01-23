from doctest import FAIL_FAST
from importlib.resources import path

from numpy import tensordot
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# version 10 : DoubleKL + txtChange(topK=5)
class Pedal(nn.Module):
    def __init__(self, scale=10, k=5,temp=0.1):
        super(Pedal, self).__init__()
        self.scale =scale
        self.k = k
        self.temp = temp
    def align_distribution(self,img_feats, text_feats):
        num_parts = img_feats.size(0)
        text_feats = text_feats.permute(1, 0, 2)  # [3,64,768]
        # 对每个局部部分计算对齐损失
        align_loss = 0
        aligned_img_list = []
        for p in range(num_parts):
            # 当前局部的图像和文本特征 [64,768]
            img_part = F.normalize(img_feats[p], p=2, dim=1)  # [64,768]
            txt_part = F.normalize(text_feats[p].detach(), p=2, dim=1)  # 完全冻结文本梯度
            # 模态内相似度矩阵
            img_sim = torch.mm(img_part, img_part.t())  # [64,64]
            txt_sim = torch.mm(txt_part, txt_part.t())  # [64,64]
            # 双向KL散度（MMT思想）
            img_log_prob = F.log_softmax(img_sim / self.temp, dim=-1)
            txt_log_prob = F.log_softmax(txt_sim / self.temp, dim=-1)
            txt_prob = F.softmax(txt_sim / self.temp, dim=-1)
            # 双向KL损失
            loss_kl = 0.5 * (
                F.kl_div(img_log_prob, txt_prob.detach(), reduction='batchmean') +
                F.kl_div(txt_log_prob.detach(), F.softmax(img_sim.detach() / self.temp, dim=-1), reduction='batchmean')
            )
            align_loss += loss_kl
            # 计算文本引导的残差项
            residual = (txt_part - img_part.detach()) * 0.1
            # 更新图像特征（保留原始梯度）
            aligned_img = img_part + residual
            aligned_img_list.append(aligned_img)
        # 合并所有局部特征 [3,64,768]
        aligned_img_feats = torch.stack(aligned_img_list, dim=0)
        return aligned_img_feats, align_loss
    def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):
        # feature 3,64,768
        # text_feature 64,3,768
        # feature = feature.permute(1, 0, 2) # 64,3,768
        # 相似度分布对齐
        aligned_img_feats, align_loss = self.align_distribution(feature,text_feature)
        feature = aligned_img_feats
        all_posvid = []
        loss = 0    

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
        loss = loss.div(feature.size(0)) + 0.5 * align_loss
        return loss, all_posvid



# version 8： 相似度靠近实验 ：双向KL散度 + 动态温度参数 +  过滤低置信度的相似度对，减少噪声样本的影响
class Pedal(nn.Module):
    def __init__(self, scale=10, k=10, temp_init=0.5, temp_min=0.01, 
                 decay_rate=0.95, sim_threshold=0.7):
        super(Pedal, self).__init__()
        self.scale = scale
        self.k = k
        self.temp = temp_init       # 初始温度
        self.temp_min = temp_min    # 最小温度
        self.decay_rate = decay_rate# 温度衰减率
        self.sim_threshold = sim_threshold  # 相似度过滤阈值
        
    def adjust_temperature(self):
        """每个epoch后调用，指数衰减温度"""
        self.temp = max(self.temp * self.decay_rate, self.temp_min)

    def align_distribution(self, img_feats, text_feats):
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
          
          # ===== 残差更新（保持不变）=====
          residual = (txt_part - img_part.detach()) * 0.1
          aligned_img = img_part + residual
          aligned_img_list.append(aligned_img)
      
      aligned_img_feats = torch.stack(aligned_img_list, dim=0)
      return aligned_img_feats, align_loss

    def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):
        # feature 3,64,768
        # text_feature 64,3,768
        # feature = feature.permute(1, 0, 2) # 64,3,768
        # 相似度分布对齐
        aligned_img_feats, align_loss = self.align_distribution(feature,text_feature)
        feature = aligned_img_feats
        all_posvid = []
        loss = 0       
        for p in range(feature.size(0)):
            part_feat = feature[p, :, :]
            part_centers = centers[p, :, :]
            m, n = part_feat.size(0), part_centers.size(0)
            # 计算当前特征（partfeat）与记忆库特征（partcenters）的欧氏距离矩阵 dist_map
            dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                       part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_map.addmm_(1, -2, part_feat, part_centers.t())
            # 排除自身作为负样本
            trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
            neg, index = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
            # 筛选跨摄像头的负样本
            pos_camid = torch.tensor(PatchMemory.camid).cuda()
            pos_camid = pos_camid[(index[:,:self.k])]
            flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)
            # 正样本选择
            pos_vid = torch.tensor(PatchMemory.vid).cuda()
            pos_vid = pos_vid[(index[:,:self.k])]
            all_posvid.append(pos_vid)
            # 使用对比损失（Contrastive Loss）形式，鼓励正样本靠近、负样本远离
            x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
            y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
            # 计算基本损失
            l = (-x + y).sum().div(feature.size(1))
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
            loss += l
        loss = loss.div(feature.size(0)) + 0.5 * align_loss
        return loss, all_posvid
    

# version 7： 相似度靠近实验 ：KL散度 + 修正后的origin（仅考虑正常跨摄像头）+ id相同->相似度得分置为1（风险）

# version 6： 相似度靠近实验 ：KL散度 + 修正后的origin（仅考虑正常跨摄像头）
# class Pedal(nn.Module):
#     def __init__(self, scale=10, k=10,temp=0.1):
#         super(Pedal, self).__init__()
#         self.scale =scale
#         self.k = k
#         self.temp = temp
#     def align_distribution(self,img_feats, text_feats):
#         num_parts = img_feats.size(0)
#         text_feats = text_feats.permute(1, 0, 2)  # [3,64,768]
#         # 对每个局部部分计算对齐损失
#         align_loss = 0
#         aligned_img_list = []
#         for p in range(num_parts):
#             # 当前局部的图像和文本特征 [64,768]
#             img_part = F.normalize(img_feats[p], p=2, dim=1)  # [64,768]
#             txt_part = F.normalize(text_feats[p].detach(), p=2, dim=1)  # 完全冻结文本梯度
#             # 模态内相似度矩阵
#             img_sim = torch.mm(img_part, img_part.t())  # [64,64]
#             txt_sim = torch.mm(txt_part, txt_part.t())  # [64,64]
#             # KL散度损失 
#             img_log_prob = F.log_softmax(img_sim / self.temp, dim=-1)
#             txt_prob = F.softmax(txt_sim / self.temp, dim=-1)
#             align_loss += F.kl_div(input=img_log_prob,target=txt_prob,reduction='batchmean')
#             # 计算文本引导的残差项
#             residual = (txt_part - img_part.detach()) * 0.1
#             # 更新图像特征（保留原始梯度）
#             aligned_img = img_part + residual
#             aligned_img_list.append(aligned_img)
#         # 合并所有局部特征 [3,64,768]
#         aligned_img_feats = torch.stack(aligned_img_list, dim=0)
#         return aligned_img_feats, align_loss
#     def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):
#         # feature 3,64,768
#         # text_feature 64,3,768
#         # feature = feature.permute(1, 0, 2) # 64,3,768
#         # 相似度分布对齐
#         aligned_img_feats, align_loss = self.align_distribution(feature,text_feature)
#         feature = aligned_img_feats
#         all_posvid = []
#         loss = 0       
#         for p in range(feature.size(0)):
#             part_feat = F.normalize(feature[p], p=2, dim=1)  # [64,768]
#             part_centers = F.normalize(centers[p], p=2, dim=1)  # [M,768]
#             batch_size = part_feat.size(0)
#             # MemoryBank中取vid和camid
#             mem_vid = torch.tensor(PatchMemory.vid, device=part_feat.device)    # [M]
#             mem_camid = torch.tensor(PatchMemory.camid, device=part_feat.device) # [M]
#             # 计算原始距离矩阵
#             dist_map = torch.cdist(part_feat, part_centers)  # [64,M]
#             # 正样本筛选（跨摄像头且相同vid）
#             same_vid_mask = (mem_vid.unsqueeze(0) == vid.unsqueeze(1))    # [64,M]
#             diff_cam_mask = (mem_camid.unsqueeze(0) != camid.unsqueeze(1))# [64,M]
#             pos_mask = same_vid_mask & diff_cam_mask
#             # 确保至少存在k个正样本
#             pos_dist, pos_idx = torch.topk(dist_map * pos_mask.float() + 1e6 * (~pos_mask).float(), k=self.k, dim=1,largest=False) #[64, k]
#             # 记录正样本vid
#             all_posvid.append(mem_vid[pos_idx])
#             # 负样本筛选 （不同vid 或 同摄像头）
#             neg_mask = (mem_vid.unsqueeze(0) != vid.unsqueeze(1)) | \
#                       (mem_camid.unsqueeze(0) == camid.unsqueeze(1))
#             neg_mask[torch.arange(batch_size), position] = False  # 排除自身
#             # 对每个样本选择固定数量的负样本
#             max_neg_per_sample = 1000
#             neg_dist, _ = torch.topk(dist_map * neg_mask.float() + 1e6 * (~neg_mask).float(), k=max_neg_per_sample, dim=1, largest=False)# [64, 1000]
#             x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # [64]，正样本项（最大化相似度）
#             y = (-self.scale * neg_dist).exp().sum(dim=1).log() # [64]，负样本项（最小化相似度）
#             # 计算基本损失
#             l = (-x + y).sum().div(feature.size(1))
#             l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
#             loss += l
#         loss = loss.div(feature.size(0)) + 0.5 * align_loss
#         return loss, all_posvid

# version 5： 相似度靠近实验 ：KL散度 + origin 
# class Pedal(nn.Module):
#     def __init__(self, scale=10, k=10,temp=0.1):
#         super(Pedal, self).__init__()
#         self.scale =scale
#         self.k = k
#         self.temp = 0.1
#     def align_distribution(self,img_feats, text_feats):
#         num_parts = img_feats.size(0)
#         text_feats = text_feats.permute(1, 0, 2)  # [3,64,768]
#         # 对每个局部部分计算对齐损失
#         align_loss = 0
#         aligned_img_list = []
#         for p in range(num_parts):
#             # 当前局部的图像和文本特征 [64,768]
#             img_part = F.normalize(img_feats[p], p=2, dim=1)  # [64,768]
#             txt_part = F.normalize(text_feats[p].detach(), p=2, dim=1)  # 完全冻结文本梯度
#             # 模态内相似度矩阵
#             img_sim = torch.mm(img_part, img_part.t())  # [64,64]
#             txt_sim = torch.mm(txt_part, txt_part.t())  # [64,64]
#             # KL散度损失 
#             img_log_prob = F.log_softmax(img_sim / self.temp, dim=-1)
#             txt_prob = F.softmax(txt_sim / self.temp, dim=-1)
#             align_loss += F.kl_div(input=img_log_prob,target=txt_prob,reduction='batchmean')
#             # 计算文本引导的残差项
#             residual = (txt_part - img_part.detach()) * 0.1
#             # 更新图像特征（保留原始梯度）
#             aligned_img = img_part + residual
#             aligned_img_list.append(aligned_img)
#         # 合并所有局部特征 [3,64,768]
#         aligned_img_feats = torch.stack(aligned_img_list, dim=0)
#         return aligned_img_feats, align_loss
#     def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):
#         # feature 3,64,768
#         # text_feature 64,3,768
#         # feature = feature.permute(1, 0, 2) # 64,3,768
#         # 相似度分布对齐
#         aligned_img_feats, align_loss = self.align_distribution(feature,text_feature)
#         feature = aligned_img_feats
#         all_posvid = []
#         loss = 0       
#         for p in range(feature.size(0)):
#             part_feat = feature[p, :, :]
#             part_centers = centers[p, :, :]
#             m, n = part_feat.size(0), part_centers.size(0)
#             # 计算当前特征（partfeat）与记忆库特征（partcenters）的欧氏距离矩阵 dist_map
#             dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
#                        part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
#             dist_map.addmm_(1, -2, part_feat, part_centers.t())
#             # 排除自身作为负样本
#             trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
#             neg, index = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
#             # 筛选跨摄像头的负样本
#             pos_camid = torch.tensor(PatchMemory.camid).cuda()
#             pos_camid = pos_camid[(index[:,:self.k])]
#             flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)
#             # 正样本选择
#             pos_vid = torch.tensor(PatchMemory.vid).cuda()
#             pos_vid = pos_vid[(index[:,:self.k])]
#             all_posvid.append(pos_vid)
#             # 使用对比损失（Contrastive Loss）形式，鼓励正样本靠近、负样本远离
#             x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
#             y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
#             # 计算基本损失
#             l = (-x + y).sum().div(feature.size(1))
#             l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
#             loss += l
#         loss = loss.div(feature.size(0)) + 0.5 * align_loss
#         return loss, all_posvid
    

# # version 4： origin + 跨域 -- return：跨域正样本
# class Pedal(nn.Module):

#     def __init__(self, scale=10, k=10, kl_weight=0.1):
#         super(Pedal, self).__init__()
#         self.scale =scale
#         self.k = k
#         self.kl_weight = kl_weight

#     def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):

#         # feature = feature.permute(1, 0, 2) # 64,3,768
        
#         batch_size = feature.size(1)
#         all_posvid = []
        
#         # 批量检索跨域索引 [batch_size, topk]
#         cross_indices = PatchMemory.retrieve(text_feature, domains, topk=5)
#         cross_indices = cross_indices.to(torch.int64)  # 确保整数类型
        
#         # 记录正样本vid [B, 5]
#         pos_vid = torch.tensor(PatchMemory.vid, device=feature.device)[cross_indices] 
#         all_posvid.append(pos_vid)
#         loss = 0      
#         for p in range(feature.size(0)):
#             part_feat = feature[p, :, :]
#             part_centers = centers[p, :, :]

#             # 跨域
#             # 获取正样本特征 [B, 5, D]
#             pos_features = part_centers[cross_indices]  # 使用跨域检索结果
#             # 计算正样本距离 [B, 5]
#             pos_dist = torch.cdist(part_feat.unsqueeze(1), pos_features).squeeze(1)
#             # 负样本筛选（排除正样本和自身）
#             # 生成负样本掩码 [B, M]
#             neg_mask = torch.ones_like(part_centers[:, 0], dtype=torch.bool)  # [M]
#             neg_mask[cross_indices.flatten()] = False  # 排除正样本
#             neg_mask[position] = False  # 排除自身
#             # 获取有效负样本距离 [B, M - 5 - 1]
#             neg_dist = torch.cdist(part_feat, part_centers[neg_mask])  # [B, N_neg]
#             # 对比损失计算
#             x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # 分子项 [B]
#             y = (-self.scale * neg_dist).exp().sum(dim=1).log()  # 分母项 [B]
#             # 计算基本损失
#             l_diff = (-x + y).sum().div(feature.size(1))
#             l_diff = torch.where(torch.isnan(l_diff), torch.full_like(l_diff, 0.), l_diff)
#             loss += l_diff
#             # origin
#             m, n = part_feat.size(0), part_centers.size(0)
#             # 计算当前特征（partfeat）与记忆库特征（partcenters）的欧氏距离矩阵 dist_map
#             dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
#                        part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
#             dist_map.addmm_(1, -2, part_feat, part_centers.t())
#             # 排除自身作为负样本
#             trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
#             neg, index = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
#             # 筛选跨摄像头的负样本
#             pos_camid = torch.tensor(PatchMemory.camid).cuda()
#             pos_camid = pos_camid[(index[:,:self.k])]
#             flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)
#             # 正样本选择
#             pos_vid = torch.tensor(PatchMemory.vid).cuda()
#             pos_vid = pos_vid[(index[:,:self.k])]
#             all_posvid.append(pos_vid)
#             # 使用对比损失（Contrastive Loss）形式，鼓励正样本靠近、负样本远离
#             x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
#             y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
#             # 计算基本损失
#             l = (-x + y).sum().div(feature.size(1))
#             l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
#             loss += l
            
#         loss = loss.div(feature.size(0))

#         return loss, all_posvid

# version 3： 跨域1 -- return： 跨域正样本
# class Pedal(nn.Module):

#     def __init__(self, scale=10, k=10, kl_weight=0.1):
#         super(Pedal, self).__init__()
#         self.scale =scale
#         self.k = k
#         self.kl_weight = kl_weight

#     def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):

#         # feature = feature.permute(1, 0, 2) # 64,3,768
        
#         batch_size = feature.size(1)
#         all_posvid = []
        
#         # 批量检索跨域索引 [batch_size, topk]
#         cross_indices = PatchMemory.retrieve(text_feature, domains, topk=5)
#         cross_indices = cross_indices.to(torch.int64)  # 确保整数类型
        
#         # 记录正样本vid [B, 5]
#         pos_vid = torch.tensor(PatchMemory.vid, device=feature.device)[cross_indices] 
#         all_posvid.append(pos_vid)
#         loss = 0      
#         for p in range(feature.size(0)):
#             part_feat = feature[p, :, :]
#             part_centers = centers[p, :, :]
#             m, n = part_feat.size(0), part_centers.size(0)
            
#             # 计算当前特征（partfeat）与记忆库特征（partcenters）的欧氏距离矩阵 dist_map
#             dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
#                        part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
#             dist_map.addmm_(1, -2, part_feat, part_centers.t())
           
#             # 排除自身作为负样本
#             trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
#             neg, index = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
            
#             # 筛选跨摄像头的负样本
#             pos_camid = torch.tensor(PatchMemory.camid).cuda()
#             pos_camid = pos_camid[(index[:,:self.k])]
#             flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)
            
          
#             # 使用对比损失（Contrastive Loss）形式，鼓励正样本靠近、负样本远离
#             x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
#             y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
#             # 计算基本损失
#             l = (-x + y).sum().div(feature.size(1))
#             l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)

#             loss += l
            
#         loss = loss.div(feature.size(0))

#         return loss, all_posvid

# version 2 ： 纯跨域2
# class Pedal(nn.Module):
#     def __init__(self, scale=10, k=10, kl_weight=0.1):
#         super(Pedal, self).__init__()
#         self.scale =scale
#         self.k = k
#         self.kl_weight = kl_weight

#     def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None,domains=None):

#         # feature = feature.permute(1, 0, 2) # 64,3,768
        
#         batch_size = feature.size(1)
#         all_posvid = []
        
#         # 1. 跨域检索获取正样本索引 [B, 5]
#         cross_indices = PatchMemory.retrieve(text_feature, domains, topk=5)
#         cross_indices = cross_indices.to(torch.int64)
        
#         # 记录正样本vid [B, 5]
#         pos_vid = torch.tensor(PatchMemory.vid, device=feature.device)[cross_indices]
#         all_posvid.append(pos_vid)
        
#         loss = 0
#         for p in range(feature.size(0)):
#             part_feat = feature[p]  # [B, D]
#             part_centers = centers[p]  # [M, D]
            
#             # 2. 获取正样本特征 [B, 5, D]
#             pos_features = part_centers[cross_indices]  # 使用跨域检索结果
            
#             # 3. 计算正样本距离 [B, 5]
#             pos_dist = torch.cdist(part_feat.unsqueeze(1), pos_features).squeeze(1)
            
#             # 4. 负样本筛选（排除正样本和自身）
#             # 生成负样本掩码 [B, M]
#             neg_mask = torch.ones_like(part_centers[:, 0], dtype=torch.bool)  # [M]
#             neg_mask[cross_indices.flatten()] = False  # 排除正样本
#             neg_mask[position] = False  # 排除自身
            
#             # 5. 获取有效负样本距离 [B, M - 5 - 1]
#             neg_dist = torch.cdist(part_feat, part_centers[neg_mask])  # [B, N_neg]
            
#             # 6. 对比损失计算
#             x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # 分子项 [B]
#             y = (-self.scale * neg_dist).exp().sum(dim=1).log()  # 分母项 [B]
#             # 计算基本损失
#             l = (-x + y).sum().div(feature.size(1))
#             l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
        
#             loss += l
            
#         loss = loss.div(feature.size(0))

#         return loss, all_posvid

# version 1 ：origin
# class Pedal(nn.Module):

#     def __init__(self, scale=10, k=10, kl_weight=0.1):
#         super(Pedal, self).__init__()
#         self.scale =scale
#         self.k = k
#         self.kl_weight = kl_weight

#     def forward(self, feature,text_feature, centers,text_centers, position, PatchMemory = None, vid=None, camid=None):

#         loss = 0      
#         all_posvid = []
#         for p in range(feature.size(0)):
#             part_feat = feature[p, :, :]
#             part_centers = centers[p, :, :]
#             m, n = part_feat.size(0), part_centers.size(0)
#             # text
#             part_text_feat = text_feature[:, p, :]
#             part_text_centers = text_centers[p, :, :]
#             # 计算当前特征（partfeat）与记忆库特征（partcenters）的欧氏距离矩阵 dist_map
#             dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
#                        part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
#             dist_map.addmm_(1, -2, part_feat, part_centers.t())
#             # text
#             dist_map_text = part_text_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
#                        part_text_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
#             dist_map_text.addmm_(1, -2, part_text_feat, part_text_centers.t())

#             # 排除自身作为负样本
#             trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
#             neg, index = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)
#             # text
#             trick_text = torch.arange(dist_map_text.size(1)).cuda().expand_as(dist_map_text)
#             neg_text, index_text = dist_map_text[trick_text!=position.unsqueeze(dim=1).expand_as(dist_map_text)].view(dist_map_text.size(0), -1).sort(dim=1)

#             # 筛选跨摄像头的负样本
#             pos_camid = torch.tensor(PatchMemory.camid).cuda()
#             pos_camid = pos_camid[(index[:,:self.k])]
#             flag = pos_camid != camid.unsqueeze(dim=1).expand_as(pos_camid)
#             # 正样本选择
#             pos_vid = torch.tensor(PatchMemory.vid).cuda()
#             pos_vid = pos_vid[(index[:,:self.k])]
#             all_posvid.append(pos_vid)
          
#             # 使用对比损失（Contrastive Loss）形式，鼓励正样本靠近、负样本远离
#             x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
            
#             y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()
#             # 计算基本损失
#             l = (-x + y).sum().div(feature.size(1))
#             l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)

#             # # text
#             # # 使用对比损失（Contrastive Loss）形式，鼓励正样本靠近、负样本远离
#             # x = ((-1 * self.scale * neg_text[:, :self.k]).exp().sum(dim=1)).log()
#             # y = ((-1 * self.scale * neg_text).exp().sum(dim=1)).log()
#             # # 计算基本损失
#             # l_text = (-x + y).sum().div(text_feature.size(1))
#             # l_text = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
            
#             loss += l
            
#         loss = loss.div(feature.size(0))

#         return loss, all_posvid
    
class CrossDomainPedal(Pedal):  # 继承原Pedal类
    def __init__(self, 
                 cross_weight=0.5, 
                 text_temp=0.1,
                 fusion_ratio=0.3):
        super().__init__()
        self.cross_weight = cross_weight
        self.text_temp = text_temp  # 文本相似度温度系数
        self.fusion_ratio = fusion_ratio  # 文本权重比例

    def forward(self, img_feats, txt_feats,patch_agent,patch_text_agent, position, memory,vid, camid, domains):
        
        # original_loss,original_all_posvid = super().forward(img_feats, txt_feats,patch_agent,patch_text_agent, position,memory,vid, camid)
        
        img_feats = img_feats.permute(1, 0, 2) # 64,3,768
        
        batch_size = img_feats.size(0)
        cross_loss = 0
        all_posvid = []
        
        # 批量检索跨域索引 [batch_size, topk]
        cross_indices = memory.retrieve(txt_feats, domains, topk=5)
        cross_indices = cross_indices.to(torch.int64)  # 确保整数类型
        
        # 记录正样本vid [B, 5]
        pos_vid = torch.tensor(memory.vid, device=img_feats.device)[cross_indices] 
        all_posvid.append(pos_vid)

        # ===== 2. 图像对比损失计算 =====
        for part_idx in range(3):
            # 当前局部特征 [B, D]
            part_feat = img_feats[:,part_idx,:]
            
            # 跨域正样本特征 [B, 5, D]
            pos_imgs = F.normalize(patch_agent[part_idx, cross_indices, :])
            
            # ===== 欧式距离计算 =====
            # 正样本距离 [B, 5]
            pos_dist = (part_feat.unsqueeze(1) - pos_imgs).pow(2).sum(dim=2)
            
            # 随机负样本 [B, 10]
            neg_indices = torch.randint(0, patch_agent.size(1), (batch_size, 10))
            neg_imgs = patch_agent[part_idx, neg_indices, :]
            neg_dist = (part_feat.unsqueeze(1) - neg_imgs).pow(2).sum(dim=2)

            # ===== 对比损失计算 =====
            x = (-self.scale * pos_dist).exp().sum(dim=1).log()  # [B]
            y = (-self.scale * neg_dist).exp().sum(dim=1).log()  # [B]
            
            l = (-x + y).sum()
            l = torch.where(torch.isnan(l), torch.full_like(l, 0.), l)
            cross_loss += l
        
        total_loss =   cross_loss / (batch_size * 3)
        return total_loss,all_posvid


# # 如果有 soft_labels，使用它进行加权
#             if soft_labels is not None:
#                 # 这里的soft_labels是形状 [B, 3]，可以与计算的损失进行逐元素相乘
#                 l = (l * soft_labels[p]).sum()
class Ipfl(nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-6, max_iter=15, nearest=3, num=2, swap=False):

        super(Ipfl, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.max_iter = max_iter
        self.num = num
        self.nearest = nearest


    def forward(self, feature, centers):

        image_label = torch.arange(feature.size(0) // self.num).repeat(self.num, 1).transpose(0, 1).contiguous().view(-1)
        center_label = torch.arange(feature.size(0) // self.num)
        loss = 0
        size = 0

        for i in range(0, feature.size(0), 1):
            label = image_label[i]
            diff = (feature[i, :].expand_as(centers) - centers).pow(self.p).sum(dim=1)
            diff = torch.sqrt(diff)

            same = diff[center_label == label]
            sorted, index = diff[center_label != label].sort()
            trust_diff_label = []
            trust_diff = []

            # cycle ranking
            max_iter = self.max_iter if self.max_iter < index.size(0) else index.size(0)
            for j in range(max_iter):
                s = centers[center_label != label, :][index[j]]
                l = center_label[center_label != label][index[j]]

                sout = (s.expand_as(centers) - centers).pow(self.p).sum(dim=1)
                sout = sout.pow(1. / self.p)

                ssorted, sindex = torch.sort(sout)
                near = center_label[sindex[:self.nearest]]
                if (label not in near):  # view as different identity
                    trust_diff.append(sorted[j])
                    trust_diff_label.append(l)
                    break

            if len(trust_diff) == 0:
                trust_diff.append(torch.tensor([0.]).cuda())

            min_diff = torch.stack(trust_diff, dim=0).min()

            dist_hinge = torch.clamp(self.margin + same.mean() - min_diff, min=0.0)

            size += 1
            loss += dist_hinge

        loss = loss / size
        return loss


class TripletHard(nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-5, swap=False, norm=False):
        super(TripletHard, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.norm = norm
        self.sigma = 3


    def forward(self, feature, label):

        if self.norm:
            feature = feature.div(feature.norm(dim=1).unsqueeze(1))
        loss = 0

        m, n = feature.size(0), feature.size(0)
        dist_map = feature.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                   feature.pow(2).sum(dim=1, keepdim=True).expand(n, m).t() + self.eps
        dist_map.addmm_(1, -2, feature, feature.t()).sqrt_()

        sorted, index = dist_map.sort(dim=1)

        for i in range(feature.size(0)):

            same = sorted[i, :][label[index[i, :]] == label[i]]
            diff = sorted[i, :][label[index[i, :]] != label[i]]
            dist_hinge = torch.clamp(self.margin + same[1] - diff.min(), min=0.0)
            loss += dist_hinge

        loss = loss / (feature.size(0))
        return loss
