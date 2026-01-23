from cmath import isnan
import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, all_posvid=None, soft_label=False, soft_weight=0.1, soft_lambda=0.2):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        # 创建标准的 one-hot 目标
        targets_one_hot = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu:
            targets_one_hot = targets_one_hot.cuda()

        # [NEW] 核心修改：仅在 soft_label=True 且 all_posvid 有效时，才计算软标签
        # 这样当 all_posvid=None 时，会自动进入 else 分支，实现逻辑退化
        if soft_label and all_posvid is not None and len(all_posvid) > 0:
            # 将所有软标签计算逻辑移动到这个安全的判断内部
            try:
                # 拼接正样本ID
                all_posvid_tensor = torch.cat(all_posvid, dim=1)

                # 计算软标签
                soft_targets_list = []
                for i in range(all_posvid_tensor.size(0)):
                    s_id, s_num = torch.unique(all_posvid_tensor[i, :], return_counts=True)
                    sum_num = s_num.sum()
                    if sum_num > 0:
                        # 使用 .long() 确保 s_id 是长整型索引
                        temp = torch.zeros(inputs.size(1)).cuda().scatter_(0, s_id.long(), (soft_lambda / sum_num) * s_num)
                        soft_targets_list.append(temp)
                    else:  # 如果某个样本没有正样本ID，则其软标签部分为0
                        soft_targets_list.append(torch.zeros(inputs.size(1)).cuda())
                
                soft_targets = torch.stack(soft_targets_list, dim=0)

                # 结合硬标签和软标签
                final_soft_targets = (1 - soft_lambda) * targets_one_hot + soft_targets
                
                # 标准标签平滑
                targets_smoothed = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
                
                # 计算最终的加权损失
                loss_standard_part = (- targets_smoothed * log_probs).mean(0).sum()
                loss_soft_part = (- final_soft_targets * log_probs).mean(0).sum()
                
                loss = loss_standard_part * (1 - soft_weight) + loss_soft_part * soft_weight

            except Exception as e:
                # 如果软标签计算过程中出现任何意外错误，则安全退回到标准损失，并打印警告
                print(f"Warning: Soft label calculation failed with error: {e}. Falling back to standard loss.")
                targets_smoothed = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
                loss = (- targets_smoothed * log_probs).mean(0).sum()

        else:
            # [NEW] 退化逻辑分支：
            # 如果 soft_label=False 或者 all_posvid 无效，则执行标准的标签平滑损失计算
            targets_smoothed = (1 - self.epsilon) * targets_one_hot + self.epsilon / self.num_classes
            loss = (- targets_smoothed * log_probs).mean(0).sum()
            
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()