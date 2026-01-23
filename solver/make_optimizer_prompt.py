import torch

def make_optimizer_prompt(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "clsctx" in key:
            lr = cfg.prompt_lr
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        if 'dcgrl' in key:
            lr = cfg.lamda
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.prompt_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer

def make_optimizer_prompt_domain(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "dmctx" in key:
            lr = cfg.prompt_lr
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        if "dc" in key and 'dcgrl' not in key:
            print(key)
            lr = cfg.lamda
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.prompt_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def make_optimizer_for_IE(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        # fix: prompt_learner -> promptlearner
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.image_encoder_lr
        weight_decay = 0.0001
        if "bias" in key:
            lr = cfg.image_encoder_lr * 2
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.image_encoder_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)

    return optimizer

def make_optimizer(model, cfg, stage):
    """
    统一优化器函数，根据训练阶段适配参数配置
    stage: "initialization" / "stage1" / "stage2"
    """
    # 1. 先根据阶段设置参数的requires_grad状态
    setup_training_stage(model, stage)
    
    params = []
    keys = []
    weight_decay = 1e-4
    
    # 2. 根据阶段收集参数并设置学习率
    if stage == "init" or stage == "stage2":
        # 初始化和第二阶段：优化image_encoder及相关组件
        # 基础学习率：使用image_encoder_lr
        base_lr = cfg.image_encoder_lr
        
        # 为不同组件设置不同的学习率
        image_encoder_params = []
        norm_params = []
        classifier_params = []
        bottleneck_params = []
        # local_proj_params = []  # 新增：局部投影层参数组
        # local_proj_proj_params = []  # 新增：局部投影层参数组
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "image_encoder" in name:
                image_encoder_params.append(param)
            elif "img_norm" in name or "img_norm_proj" in name:
                norm_params.append(param)
            elif "classifier" in name:
                classifier_params.append(param)
            elif "bottleneck" in name:
                bottleneck_params.append(param)
            elif "local_proj." in name:  # 新增：收集局部投影层参数
                local_proj_params.append(param)
            elif "local_proj_proj." in name:  # 新增：收集局部投影层参数
                local_proj_proj_params.append(param)
        
        # 添加各组件参数组，设置不同学习率倍率
        if image_encoder_params:
            params.append({
                "params": image_encoder_params,
                "lr": base_lr,
                "weight_decay": weight_decay
            })
        if norm_params:
            params.append({
                "params": norm_params,
                "lr": base_lr * 1.0,
                "weight_decay": weight_decay
            })
        if classifier_params:
            params.append({
                "params": classifier_params,
                "lr": base_lr * 1.2,  # 分类器学习率略高
                "weight_decay": weight_decay
            })
        if bottleneck_params:
            params.append({
                "params": bottleneck_params,
                "lr": base_lr * 1.0,
                "weight_decay": weight_decay
            })
        # if local_proj_params:
        #     params.append({
        #         "params": local_proj_params,
        #         "lr": base_lr * 1.5,  # 局部投影层学习率更高
        #         "weight_decay": weight_decay
        #     })
        # if local_proj_proj_params:
        #     params.append({
        #         "params": local_proj_proj_params,
        #         "lr": base_lr * 1.5,
        #         "weight_decay": weight_decay
        #     })

    elif stage == "stage1":
        # 第一阶段：优化promptlearner和dcgrl
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # promptlearner.clsctx参数：使用prompt_lr
            if "promptlearner" in name:
                lr = cfg.prompt_lr
                params.append({
                    "params": [param],
                    "lr": lr,
                    "weight_decay": weight_decay
                })
                keys.append(name)
    
    # 3. 根据配置选择优化器类型
    if cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, momentum=0.95)
    elif cfg.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, weight_decay=weight_decay)  # 不指定全局lr
    else:
        optimizer = torch.optim.Adam(params)
    
    return optimizer

def setup_training_stage(model, stage):
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 初始化阶段：训练image_encoder及相关组件
    if stage == "init":
        for name, param in model.named_parameters():
            if (
                "image_encoder" in name or
                "img_norm" in name or
                "img_norm_proj" in name or  # 新增：包含投影层的归一化
                "classifier" in name or
                "bottleneck" in name or
                "local_proj" in name or  # 新增：包含局部投影层
                "local_proj_proj" in name  # 新增：包含局部投影层
            ):
                param.requires_grad = True
    
    # 第一阶段：只训练promptlearner
    elif stage == "stage1":
        for name, param in model.named_parameters():
            if (
                "promptlearner" in name
            ):
                param.requires_grad = True
    
    # 第二阶段：微调image_encoder及相关组件
    elif stage == "stage2":
        for name, param in model.named_parameters():
            if (
                "image_encoder" in name or
                "img_norm" in name or
                "img_norm_proj" in name or  # 新增：包含投影层的归一化
                "classifier" in name or
                "bottleneck" in name or
                "local_proj" in name or  # 新增：包含局部投影层
                "local_proj_proj" in name  # 新增：包含局部投影层
            ):
                param.requires_grad = True
            
            # # 选择性训练text_encoder的顶层
            # if "text_encoder" in name and ("transformer.resblocks.10" in name or "transformer.resblocks.11" in name):
            #     param.requires_grad = True  # 只训练最后两层 (ViT-B-16有12层)