from collections import defaultdict
import torch
import torch.nn.functional as F


class PatchMemory(object):

    def __init__(self, momentum=0.1, num=1):

        self.name = []
        self.agent = []
        self.text_agent = []  # text MemoryBank 
        self.momentum = momentum
        self.num = num
        
        self.camid = []
        self.vid = []

        self.domains = torch.tensor([], dtype=torch.long)
        
    def get_soft_label(self, path, feat_list,text_embeddings,domains, vid=None, camid=None):
        # 为每个样本的每个局部特征生成域标签 [B×3]
        expanded_domains = []
        for d in domains:  # domains形状应为[B]
            expanded_domains.extend([d] * 3)  
        
        new_domain_indices = []
        
        feat = torch.stack(feat_list, dim=0)
        feat = feat[:, ::self.num, :]
        position = []

        text_embeddings = text_embeddings[:, 1:4, :]  # part text feat
        text_embeddings = text_embeddings[:, ::self.num, :]

        # update the agent
        for j,p in enumerate(path):
            current_soft_text = text_embeddings[j, :, :].detach()
            current_soft_feat = feat[:, j, :].detach()
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()
            if current_soft_text.is_cuda:
                current_soft_text = current_soft_text.cpu()    
            key  = p
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
                tmp = tmp*(1-self.momentum) + self.momentum*current_soft_feat
                tmp1 = self.text_agent.pop(ind)
                tmp1 = tmp1*(1-self.momentum) + self.momentum*current_soft_text
                self.agent.insert(ind, tmp)
                self.text_agent.insert(ind, tmp1)
                position.append(ind)

        # 更新域标签库（仅添加新样本的域标签）
        if new_domain_indices:
            # 获取新样本对应的域标签 [新样本数×3]
            new_domains = torch.tensor(
                [expanded_domains[i*3] for i in new_domain_indices],  # 每个样本取一个域标签
                dtype=torch.long,
                device=self.domains.device
            )
            self.domains = torch.cat([self.domains, new_domains], dim=0)

        if len(position) != 0:
            position = torch.tensor(position).cuda()
    
        agent = torch.stack(self.agent, dim=1).cuda()
        text_agent = torch.stack(self.text_agent, dim=1).cuda()
      
        return agent, text_agent,position

    def retrieve(self, query_txt ,current_domain, topk=5):
        # 将内存库中的特征转换为Tensor
        text_agent = torch.stack(self.text_agent, dim=0)  # [M, num_parts, D]
        query_txt = query_txt.to(text_agent.device)
        M = text_agent.size(0) * 3
        domains_reshaped = self.domains.view(-1)
        # 跨域mask [B, M]
        cross_mask = (domains_reshaped.unsqueeze(0) != current_domain.unsqueeze(1))  # [B, M]
        # 多局部相似度融合
        sim_all = []
        for part_idx in range(3):
            # 获取当前局部特征
            part_query = query_txt[:, part_idx, :]  # [B, D]
            part_memory = text_agent[:, part_idx, :]  # [M, D]
            
            # 计算相似度 [B, M]
            sim = torch.mm(
                F.normalize(part_query, dim=1),
                F.normalize(part_memory, dim=1).t()
            )
            sim_all.append(sim.unsqueeze(1))  # [B, 1, M]
        
        # 聚合三个局部的相似度
        total_sim = torch.cat(sim_all, dim=1).mean(dim=1)  # [B, M]
        total_sim.masked_fill_(~cross_mask, -float('inf'))
        
        # 获取Top-K索引
        _, topk_indices = total_sim.topk(k=topk, dim=1)
        return topk_indices
    
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



class SmoothingForImage(object):
    def __init__(self, momentum=0.1, num=1):

        self.map = dict()
        self.momentum = momentum
        self.num = num


    def get_soft_label(self, path, feature):

        feature = torch.cat(feature, dim=1)
        soft_label = []

        for j,p in enumerate(path):

            current_soft_feat = feature[j*self.num:(j+1)*self.num, :].detach().mean(dim=0)
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()

            key  = p
            if key not in self.map:
                self.map.setdefault(key, current_soft_feat)
                soft_label.append(self.map[key])
            else:
                self.map[key] = self.map[key]*(1-self.momentum) + self.momentum*current_soft_feat
                soft_label.append(self.map[key])
        soft_label = torch.stack(soft_label, dim=0).cuda()
        return soft_label



