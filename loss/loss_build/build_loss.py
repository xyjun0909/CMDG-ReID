import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .ce_labelSmooth import CrossEntropyLabelSmooth as CE_LS

feat_dim_dict = {
    "local_attention_vit": 768,
    "vit": 768,
    "resnet18": 512,
    "resnet34": 512,
}


def build_loss(num_classes):
    name = "part_attention_vit"  
    sampler = "softmax_triplet"  
    metric_loss_type = "triplet" 
    label_smooth = "on" 

    feat_dim = feat_dim_dict[name] if name in feat_dim_dict else 2048
    center_criterion = CenterLoss(
        num_classes=num_classes,
        feat_dim=feat_dim,
        use_gpu=True
    )

    triplet = TripletLoss()
    print("using soft triplet loss for training")

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)

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
        ID_LOSS = xent(
            score,
            target,
            all_posvid=all_posvid,
            soft_label=soft_label,
            soft_weight=soft_weight,
            soft_lambda=soft_lambda,
        )
        
        TRI_LOSS = triplet(feat, target)[0]
        return 1.5 * ID_LOSS + 1 * TRI_LOSS

    return loss_func, center_criterion