import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(num_classes):
    sampler = 'softmax_triplet'
    feat_dim = 2048
    metric_loss_type = 'triplet'
    label_smooth = 'on'
    triplet_margin = 0.3

    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        use_gpu=True
    )

    # 初始化三元组损失（带margin=0.3）
    triplet = TripletLoss(triplet_margin)
    print(f"using triplet loss with margin:{triplet_margin}")

    # 初始化带标签平滑的交叉熵损失
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print(f"label smooth on, numclasses:{num_classes}")

    # 仅保留softmax_triplet分支的损失计算逻辑
    def loss_func(score, feat, target, target_cam, i2tscore=None):
        # 计算ID损失（带标签平滑）
        if isinstance(score, list):
            ID_LOSS = sum([xent(scor, target) for scor in score[0:]])
        else:
            ID_LOSS = xent(score, target)

        # 计算三元组损失
        if isinstance(feat, list):
            TRI_LOSS = sum([triplet(feats, target)[0] for feats in feat[0:]])
        else:
            TRI_LOSS = triplet(feat, target)[0]

        # 基础损失：ID损失（权重0.25）+ 三元组损失（权重1.0）
        loss = 0.25 * ID_LOSS + 1.0 * TRI_LOSS

        # 若提供i2tscore，加入图像-文本跨模态损失（权重1.0）
        if i2tscore is not None:
            I2TLOSS = xent(i2tscore, target)
            loss = 1.0 * I2TLOSS + loss

        return loss

    return loss_func, center_criterion