import torch
import torch.nn.functional as F


class PatchMemory(object):

    def __init__(self, momentum=0.1, num=1):
        self.name = []
        self.agent = []
        self.text_agent = []  # text MemoryBank
        self.momentum = momentum
        self.num = num
        self.device = 'cuda'

        self.camid = []
        self.vid = []

        self.domains = torch.tensor([], dtype=torch.long)

    def get_soft_label(self, path, img_feats, cap_feats, domains, vid=None, camid=None):
        # 为每个样本的每个局部特征生成域标签 [B×3]
        expanded_domains = []
        for d in domains:  # domains形状应为[B]
            expanded_domains.extend([d] * 3)

        new_domain_indices = []
        position = []

        # 2. 处理图像特征（前3个局部特征）
        img_feat_tensor = torch.stack(img_feats[:3], dim=0).to(self.device)  # [3, batch_size, 512]
        img_feat_tensor = img_feat_tensor[:, :: self.num, :]  # 采样

        # 3. 处理文本特征（前3个局部特征）
        cap_feat_tensor = torch.stack(cap_feats[:3], dim=0).to(self.device)  # [3, batch_size, 512]
        cap_feat_tensor = cap_feat_tensor[:, :: self.num, :]  # 采样

        # update the agent
        for j, p in enumerate(path):
            current_soft_text = cap_feat_tensor[:, j, :].detach()
            current_soft_feat = img_feat_tensor[:, j, :].detach()
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()
            if current_soft_text.is_cuda:
                current_soft_text = current_soft_text.cpu()
            key = p
            if key not in self.name:
                self.name.append(key)
                self.camid.append(camid[j])
                self.vid.append(vid[j])
                self.agent.append(current_soft_feat)
                self.text_agent.append(current_soft_text)
                ind = self.name.index(key)
                position.append(ind)

                # 处理新样本的domains
                new_domain_indices.append(j)
            else:
                ind = self.name.index(key)
                tmp = self.agent.pop(ind)
                tmp = tmp * (1 - self.momentum) + self.momentum * current_soft_feat
                tmp1 = self.text_agent.pop(ind)
                tmp1 = tmp1 * (1 - self.momentum) + self.momentum * current_soft_text
                self.agent.insert(ind, tmp)
                self.text_agent.insert(ind, tmp1)
                position.append(ind)

        # 更新域标签库（仅添加新样本的域标签）
        if new_domain_indices:
            # 获取新样本对应的域标签 [新样本数×3]
            new_domains = torch.tensor(
                [
                    expanded_domains[i * 3] for i in new_domain_indices
                ],  # 每个样本取一个域标签
                dtype=torch.long,
                device=self.domains.device,
            )
            self.domains = torch.cat([self.domains, new_domains], dim=0)

        if len(position) != 0:
            position = torch.tensor(position).cuda()

        agent = torch.stack(self.agent, dim=1).cuda()
        text_agent = torch.stack(self.text_agent, dim=1).cuda()

        return agent, text_agent, position

    def retrieve(self, query_txt, current_domain, topk=5):
        # 调整query_txt维度: [3, 64, 512] -> [64, 3, 512]
        query_txt = query_txt.permute(1, 0, 2)  # [64, 3, 512]

        # 获取内存库特征
        text_agent = torch.stack(self.text_agent, dim=0)  # [M, 3, 512]
        text_agent = text_agent.to(query_txt.device)
        M = text_agent.size(0)  # 内存库样本数 (M)

        # 确保域标签与内存库大小一致
        if self.domains.size(0) != M:
            # 如果域标签数不匹配，截取匹配的部分
            domains_reshaped = self.domains[:M].to(text_agent.device)
        else:
            domains_reshaped = self.domains.to(text_agent.device)

        current_domain = current_domain.to(text_agent.device)  # [64]

        # 创建跨域掩码: [64, M]
        cross_mask = current_domain.unsqueeze(1) != domains_reshaped.unsqueeze(0)

        # 多局部相似度融合
        sim_all = []
        for part_idx in range(3):
            # 局部特征维度: [64, 512] 和 [M, 512]
            part_query = query_txt[:, part_idx, :]  # [64, 512]
            part_memory = text_agent[:, part_idx, :]  # [M, 512]

            # 计算余弦相似度: [64, M]
            sim = torch.mm(
                F.normalize(part_query, dim=1), F.normalize(part_memory, dim=1).t()
            )
            sim_all.append(sim.unsqueeze(1))  # [64, 1, M]

        # 聚合相似度: [64, M]
        total_sim = torch.cat(sim_all, dim=1).mean(dim=1)

        # 应用跨域掩码
        total_sim = total_sim.masked_fill(~cross_mask, -float("inf"))

        # 获取Top-K索引
        _, topk_indices = total_sim.topk(k=topk, dim=1)
        return topk_indices
    
    def retrieve_all_domains(self, query_txt, topk=5):
        # 调整query_txt维度: [3, batch_size, 512] → [batch_size, 3, 512]
        query_txt = query_txt.permute(1, 0, 2)  # [batch_size, 3, 512]

        # 获取内存库中的文本特征（所有域，不区分）
        text_agent = torch.stack(self.text_agent, dim=0)  # [M, 3, 512]，M为内存库总样本数
        text_agent = text_agent.to(query_txt.device)  # 转移到与查询文本相同的设备

        # 多局部特征相似度融合（与原逻辑一致，但不过滤域）
        sim_all = []
        for part_idx in range(3):  # 对3个局部特征分别计算相似度
            # 提取第part_idx个局部的查询特征和内存库特征
            part_query = query_txt[:, part_idx, :]  # [batch_size, 512]
            part_memory = text_agent[:, part_idx, :]  # [M, 512]

            # 计算余弦相似度（归一化后矩阵乘法）
            sim = torch.mm(
                F.normalize(part_query, dim=1),  #  query特征归一化
                F.normalize(part_memory, dim=1).t()  # 内存库特征归一化并转置
            )  # 输出形状：[batch_size, M]
            sim_all.append(sim.unsqueeze(1))  # 保留维度，方便后续拼接

        # 聚合3个局部的相似度（取平均）
        total_sim = torch.cat(sim_all, dim=1).mean(dim=1)  # [batch_size, M]

        # 对所有文本（不区分域）按相似度排序，取topk
        _, topk_indices = total_sim.topk(k=topk, dim=1)  # [batch_size, topk]

        return topk_indices

    def retrieve_image_only(self, query_local_feats, query_position_in_memory, topk=5):
        """
        【最终修正版】
        精确复刻纯图像Pedal loss的检索逻辑：基于所有局部特征的欧氏距离。
        
        query_local_feats: 当前查询样本的局部特征 [3, D]
        query_position_in_memory: 当前查询样本在内存库中的索引 (一个整数)
        """
        # 获取内存库中的局部图像特征 [M, 3, D]
        # 注意：self.agent中每个元素是[3, D]，所以stack后的维度是[M, 3, D]
        image_agent = torch.stack(self.agent, dim=0).to(query_local_feats.device)

        # 用于累加所有局部特征计算出的距离
        total_dist_map = 0

        # 遍历3个局部特征，计算并累加距离矩阵
        for p in range(query_local_feats.size(0)): # 遍历 p = 0, 1, 2
            part_feat = query_local_feats[p].unsqueeze(0)  # [1, D]
            part_centers = image_agent[:, p, :]            # [M, D]
            
            # --- start: 完全复刻您的dist_map计算逻辑 ---
            m, n = part_feat.size(0), part_centers.size(0)
            dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                      part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_map.addmm_(1, -2, part_feat, part_centers.t())
            # --- end: dist_map计算逻辑结束 ---
            
            total_dist_map += dist_map

        # 对所有局部的距离求平均，得到最终的距离排序依据
        avg_dist_map = total_dist_map / query_local_feats.size(0) # [1, M]

        # --- start: 完全复刻您的排序逻辑 ---
        # 排除查询样本自身
        full_indices = torch.arange(avg_dist_map.size(1), device=avg_dist_map.device)
        mask = (full_indices != query_position_in_memory)
        
        # 对所有非自身的样本按距离进行排序 (距离越小越靠前)
        _, sorted_indices_of_negatives = avg_dist_map[0, mask].sort(descending=False)
        
        # 从内存库的完整索引中，选出这些排好序的负样本
        original_indices = full_indices[mask][sorted_indices_of_negatives]
        # --- end: 排序逻辑结束 ---
        
        # 返回前K个最相似的样本的索引
        return original_indices[:topk]