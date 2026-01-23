import os.path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        # {783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, data in enumerate(self.data_source):
            self.index_dic[data[1]].append(index)
        self.pids = list(self.index_dic.keys())
        print('pids', len(self.pids))
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        pids = self.pids
        for pid in pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        print(len(final_idxs))
        return iter(final_idxs)

    def __len__(self):
        return self.length

class DomainAwareIdentitySampler(Sampler):
    """
    域感知身份采样器：
    1. 保证每个批次包含多个域的样本，避免域分布偏差
    2. 每个身份在批次中保持固定数量的实例（num_instances）
    3. 样本不足时优先从同身份的其他域补充，减少重复采样
    """
    def __init__(self, data_source, batch_size, num_instances):
        super().__init__(data_source)
        self.data_source = data_source  # 训练样本列表，每个元素包含(domain, pid, ...)
        self.batch_size = batch_size
        self.num_instances = num_instances  # 每个身份的实例数（如4）
        self.num_pids_per_batch = self.batch_size // self.num_instances  # 每个批次的身份数
        
        # 1. 构建核心映射：域→身份→样本索引（适配你的数据结构）
        self.domain_pid_index = defaultdict(lambda: defaultdict(list))  # domain -> pid -> [indices]
        self.pid_all_domains = defaultdict(list)  # pid -> 所有域的样本索引（跨域补充用）
        for idx, item in enumerate(self.data_source):
            # 从data_source中提取域和身份（根据你的数据结构：item包含(domain, pid, ...)）
            # 对应collate_fn中的domains来源于item[4]，pid来源于item[1]
            domain = item[3]  # 域标签
            pid = item[1]     # 身份标签
            self.domain_pid_index[domain][pid].append(idx)
            self.pid_all_domains[pid].append(idx)  # 记录该身份在所有域的样本
        
        self.domains = list(self.domain_pid_index.keys())  # 所有域
        self.num_domains = len(self.domains)  # 域数量
        self.length = self._calculate_length()  # 计算epoch长度

    def _calculate_length(self):
        """估算每个epoch的样本数，确保每个域的身份被充分覆盖"""
        total = 0
        for domain in self.domains:
            for pid, indices in self.domain_pid_index[domain].items():
                num_samples = len(indices)
                # 若样本数不足num_instances，按num_instances计算（需补充）
                if num_samples < self.num_instances:
                    num_samples = self.num_instances
                total += num_samples - (num_samples % self.num_instances)
        return total

    def __iter__(self):
        # 为每个域的每个身份准备批次（每个批次含num_instances个样本）
        domain_batch_buffer = defaultdict(lambda: defaultdict(list))  # domain -> pid -> [batch_indices]
        for domain in self.domains:
            for pid, indices in self.domain_pid_index[domain].items():
                indices = copy.deepcopy(indices)
                # 若当前域的样本不足，优先从同身份的其他域补充（减少重复采样）
                if len(indices) < self.num_instances:
                    # 从其他域的同身份样本中补充
                    other_domain_indices = [idx for idx in self.pid_all_domains[pid] if idx not in indices]
                    if other_domain_indices:
                        need = self.num_instances - len(indices)
                        indices += random.sample(other_domain_indices, min(need, len(other_domain_indices)))
                # 若仍不足，才进行重复采样（最后手段）
                if len(indices) < self.num_instances:
                    indices = np.random.choice(indices, size=self.num_instances, replace=True)
                # 打乱并分组为多个批次（每个批次含num_instances个样本）
                random.shuffle(indices)
                for i in range(0, len(indices), self.num_instances):
                    batch = indices[i:i+self.num_instances]
                    if len(batch) == self.num_instances:  # 确保批次完整
                        domain_batch_buffer[domain][pid].append(batch)
        
        final_indices = []
        # 循环采样直到批次耗尽
        while True:
            # 每次采样覆盖尽可能多的域（至少2个，最多num_domains个）
            num_selected_domains = min(max(2, random.randint(1, self.num_domains)), self.num_domains)
            selected_domains = random.sample(self.domains, num_selected_domains)
            
            # 为每个选中的域分配身份名额（平均分配+余数）
            base_pids_per_domain = self.num_pids_per_batch // num_selected_domains
            remaining_pids = self.num_pids_per_batch % num_selected_domains
            domain_pid_counts = {d: base_pids_per_domain + (1 if i < remaining_pids else 0) 
                                for i, d in enumerate(selected_domains)}
            
            # 收集每个域的候选身份
            batch_pids = []
            for domain in selected_domains:
                # 该域可用的身份（还有剩余批次的）
                available_pids = [pid for pid in domain_batch_buffer[domain] if len(domain_batch_buffer[domain][pid]) > 0]
                if not available_pids:
                    continue
                # 按分配的名额采样身份
                take = min(domain_pid_counts[domain], len(available_pids))
                batch_pids.extend(random.sample(available_pids, take))
            
            # 若收集的身份数不足，退出循环
            if len(batch_pids) < self.num_pids_per_batch:
                break
            
            # 提取这些身份的样本索引，加入最终列表
            for pid in batch_pids[:self.num_pids_per_batch]:  # 确保数量正确
                # 找到该身份所属的域（简化：假设优先从选中的域中取）
                for domain in selected_domains:
                    if pid in domain_batch_buffer[domain] and len(domain_batch_buffer[domain][pid]) > 0:
                        final_indices.extend(domain_batch_buffer[domain][pid].pop(0))
                        break
        
        return iter(final_indices)

    def __len__(self):
        return self.length



