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
        expanded_domains = []
        for d in domains: 
            expanded_domains.extend([d] * 3)

        new_domain_indices = []
        position = []

        img_feat_tensor = torch.stack(img_feats[:3], dim=0).to(self.device)  # [3, batch_size, 512]
        img_feat_tensor = img_feat_tensor[:, :: self.num, :]  

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

        if new_domain_indices:
            new_domains = torch.tensor(
                [
                    expanded_domains[i * 3] for i in new_domain_indices
                ],  
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
        query_txt = query_txt.permute(1, 0, 2)  # [64, 3, 512]

       
        text_agent = torch.stack(self.text_agent, dim=0)  # [M, 3, 512]
        text_agent = text_agent.to(query_txt.device)
        M = text_agent.size(0)  

        if self.domains.size(0) != M:
            domains_reshaped = self.domains[:M].to(text_agent.device)
        else:
            domains_reshaped = self.domains.to(text_agent.device)

        current_domain = current_domain.to(text_agent.device)  # [64]

        cross_mask = current_domain.unsqueeze(1) != domains_reshaped.unsqueeze(0)

        sim_all = []
        for part_idx in range(3):
            part_query = query_txt[:, part_idx, :]  # [64, 512]
            part_memory = text_agent[:, part_idx, :]  # [M, 512]

            sim = torch.mm(
                F.normalize(part_query, dim=1), F.normalize(part_memory, dim=1).t()
            )
            sim_all.append(sim.unsqueeze(1))  # [64, 1, M]

        total_sim = torch.cat(sim_all, dim=1).mean(dim=1)

        total_sim = total_sim.masked_fill(~cross_mask, -float("inf"))

        _, topk_indices = total_sim.topk(k=topk, dim=1)
        return topk_indices
    
    def retrieve_all_domains(self, query_txt, topk=5):
        query_txt = query_txt.permute(1, 0, 2)  # [batch_size, 3, 512]

        text_agent = torch.stack(self.text_agent, dim=0)  # [M, 3, 512]，
        text_agent = text_agent.to(query_txt.device)  

        sim_all = []
        for part_idx in range(3):  
            part_query = query_txt[:, part_idx, :]  # [batch_size, 512]
            part_memory = text_agent[:, part_idx, :]  # [M, 512]

            sim = torch.mm(
                F.normalize(part_query, dim=1), 
                F.normalize(part_memory, dim=1).t()  
            )  
            sim_all.append(sim.unsqueeze(1)) 

        total_sim = torch.cat(sim_all, dim=1).mean(dim=1)  # [batch_size, M]

        _, topk_indices = total_sim.topk(k=topk, dim=1)  # [batch_size, topk]

        return topk_indices

    def retrieve_image_only(self, query_local_feats, query_position_in_memory, topk=5):
        image_agent = torch.stack(self.agent, dim=0).to(query_local_feats.device)

        total_dist_map = 0

        for p in range(query_local_feats.size(0)):
            part_feat = query_local_feats[p].unsqueeze(0)  # [1, D]
            part_centers = image_agent[:, p, :]            # [M, D]
            
            m, n = part_feat.size(0), part_centers.size(0)
            dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                      part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_map.addmm_(1, -2, part_feat, part_centers.t())
            
            total_dist_map += dist_map

        avg_dist_map = total_dist_map / query_local_feats.size(0) # [1, M]

    
        full_indices = torch.arange(avg_dist_map.size(1), device=avg_dist_map.device)
        mask = (full_indices != query_position_in_memory)
        
        _, sorted_indices_of_negatives = avg_dist_map[0, mask].sort(descending=False)
        
        original_indices = full_indices[mask][sorted_indices_of_negatives]
        
        return original_indices[:topk]