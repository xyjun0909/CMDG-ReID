import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS

# 特征维度字典，仅保留实际用到的模型对应维度
feat_dim_dict = {
    "local_attention_vit": 768,
    "vit": 768,
    "resnet18": 512,
    "resnet34": 512,
}


def build_loss(num_classes):
    # 硬编码参数（原代码中实际生效的配置）
    name = "part_attention_vit"  # 固定模型名称
    sampler = "softmax_triplet"  # 固定采样器类型
    metric_loss_type = "triplet"  # 固定损失类型
    label_smooth = "on"  # 固定开启标签平滑

    # 确定特征维度
    feat_dim = feat_dim_dict[name] if name in feat_dim_dict else 2048
    # 初始化中心损失（实际未在有效分支中使用，仍保留以匹配返回值）
    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        use_gpu=True
    )

    # 初始化三元组损失（硬编码使用软三元组损失，无margin）
    triplet = TripletLoss()
    print("using soft triplet loss for training")

    # 初始化标签平滑交叉熵损失
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)

    # 仅保留生效的softmax_triplet分支
    def loss_func(
        score,
        feat,
        target,
        i2tscore=None,
        domains=None,
        t_domains=None,
        all_posvid=None,
        soft_label=False,
        soft_weight=0.1,
        soft_lambda=0.2,
    ):
        # 计算ID损失（带标签平滑）
        ID_LOSS = xent(
            score,
            target,
            all_posvid=all_posvid,
            soft_label=soft_label,
            soft_weight=soft_weight,
            soft_lambda=soft_lambda,
        )
        
        # 计算三元组损失
        TRI_LOSS = triplet(feat, target)[0]
        return 1.5 * ID_LOSS + 1 * TRI_LOSS

    return loss_func, center_criterion