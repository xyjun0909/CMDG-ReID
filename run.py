import os
import time
import argparse
import warnings
import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import loss as Patchloss
import torch.nn.functional as F
import random

from loss.loss_build.build_loss import build_loss
from reidutils.meter import AverageMeter
from reidutils.metrics import R1_mAP_eval
from cfgs import *
from reidutils import setup_logger
import datetime
from models import *
from functools import partial
from torch.cuda import amp
from torch import nn
from datasets.build import build_data_loader
from loss import make_loss
from solver import (
    create_scheduler,
    WarmupMultiStepLR,
    make_optimizer_for_IE,
    make_optimizer_prompt,
    setup_training_stage,
    make_optimizer,
)
import sys
from loss.supcontrast import SupConLoss

import shutil
from PIL import Image, ImageDraw, ImageFont

sys.path.append("/")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_model(args):
    if args.model == "clip":
        return Clip(
            sum(args.classes),
            args,
            domain_num=len(args.train_datasets),
            epsilon=args.epsilon,
        ).cuda()
    elif args.model == "ViT":
        model = ViT(
            img_size=args.size_train,
            stride_size=16,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            qkv_bias=True,
        )
        model.load_param(args.pretrain_vit_path)
    elif args.model == "gfnet":
        model = GFNet(
            img_size=(384, 128),
            patch_size=16,
            embed_dim=384,
            depth=19,
            mlp_ratio=4,
            drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        model.load_param(args.pretrain_gfnet_path)
    else:
        model = ViT(
            img_size=args.size_train,
            stride_size=16,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            qkv_bias=True,
        )
        model.load_param(args.pretrain_path)
    return model.cuda()

def init_image_encoder(
    train_loader_stage2,
    model,
    criterion,
    optimizer,
    scheduler,
    args_train,
    logger_train,
    log_path,
    epochs=3,
):
    logger_train.info("start image encoder initialization")
    device = "cuda"
    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter.reset()
        scheduler.step()
        for n_iter, (img, vid, _, _, _, _, _, _) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            img, target = img.to(device), vid.to(device)
            with amp.autocast(enabled=True):
                score, feat, _ = model(x=img, label=target)
                loss = criterion(score, feat, target, None)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), img.shape[0])
            torch.cuda.synchronize()

            if (n_iter + 1) % args_train.log_period == 0:
                logger_train.info(
                    f"Image-Init-Epoch[{epoch}] Iteration[{n_iter+1}/{len(train_loader_stage2)}] Loss: {loss_meter.avg:.3f}, Base Lr: {scheduler.get_lr()[0]:.2e}"
                )

        epoch_losses.append(loss_meter.avg)
        torch.save(
            model.state_dict(),
            os.path.join(
                log_path,
                f"{args_train.model}_{datetime.datetime.now()}_image_init_{epoch}.pth",
            ),
        )

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker="o", label="Total Loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss Value"), plt.title("Initialization Stage Loss Curve")
    plt.grid(True), plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(log_path, "initialization_loss_curve.png"))
    plt.close()
    logger_train.info(f"Initialization loss curve saved to {log_path}")


def train_stage1(
    train_loader_stage1,
    model,
    optimizer,
    scheduler,
    args_train,
    logger_train,
    log_path,
    get_domain=False,
    epochs=120,
    omega=0.01,
):
    logger_train.info("start stage-1 training")
    device = "cuda"
    loss_meter = AverageMeter()
    accd_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss("cuda")
    dc = nn.CrossEntropyLoss()
    epoch_losses = []

    image_features, labels_list, domains_list, cids_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for _, (img, pids, camid, _, domain, cid, _, _) in enumerate(
            train_loader_stage1
        ):
            img, target = img.to(device), pids.to(device)
            img_feats_list = model(img, target, get_image=True)
            img_feats_tensor = torch.stack(img_feats_list, dim=0)

            for j in range(img.shape[0]):
                labels_list.append(target[j].cpu())
                domains_list.append(domain[j].cpu())
                cids_list.append(cid[j].cpu())
                image_features.append(img_feats_tensor[:, j, :].cpu())

    labels = torch.stack(labels_list, 0).cuda()
    domains = torch.stack(domains_list, 0).cuda()
    cids = torch.stack(cids_list, 0).cuda()
    image_feats = torch.stack(image_features, 0).cuda() 
    num_image, batch = labels.shape[0], args_train.batch_size
    i_ter = num_image // batch

    del image_features, labels_list, domains_list, cids_list

    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter.reset()
        accd_meter.reset()
        scheduler.step(epoch)
        iter_list = torch.arange(num_image).to(device)

        for i in range(i_ter):
            b_list = (
                iter_list[i * batch : (i + 1) * batch]
                if i != i_ter
                else iter_list[i * batch :]
            )
            target, domain, cid = labels[b_list], domains[b_list], cids[b_list]

            optimizer.zero_grad()
            with amp.autocast(enabled=True):
                text_feats, _ = model(
                    label=target,
                    get_text=True,
                    domain=domain,
                    cam_label=cid,
                    getdomain=get_domain,
                )
                global_img_feats = image_feats[b_list, 3, :] 
                loss_i2t = xent(
                    global_img_feats.float(), text_feats.float(), target, target
                )
                loss_t2i = xent(
                    text_feats.float(), global_img_feats.float(), target, target
                )
                loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(loss.item(), global_img_feats.shape[0])
            torch.cuda.synchronize()

            if (i + 1) % args_train.log_period == 0:
                logger_train.info(
                    f"STAGE1-Epoch[{epoch}] Iteration[{i+1}/{len(train_loader_stage1)}] Loss: {loss_meter.avg:.3f}, Lr: {scheduler._get_lr(epoch)[0]:.2e}")
            
        epoch_losses.append(loss_meter.avg)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker="o", label="Contrastive Loss (i2t + t2i)")
    plt.xlabel("Epoch"), plt.ylabel("Loss Value"), plt.title("Stage1 Loss Curve")
    plt.grid(True), plt.legend(), plt.tight_layout()
    plt.savefig(os.path.join(log_path, "stage1_loss_curve.png"))
    plt.close()
    logger_train.info(f"Stage1 loss curve saved to {log_path}")

    torch.save(
        model.promptlearner.state_dict(),
        os.path.join(log_path, f"{args_train.model}_clsctx_{str(get_domain)}.pth"),
    )

def train_stage2(
    train_loader_stage2,
    model,
    criterion,
    optimizer_image,
    scheduler_image,
    testloaders,
    args_train,
    logger_train,
    logger_test,
    log_path,
    loss_fn,
    epochs=60,
    gamma=1,
):
    logger_train.info("start stage-2 training and centers initialization")
    device = "cuda"
    loss_total_meter = AverageMeter()  
    loss_reid_meter = AverageMeter()  
    loss_partcontrastive_meter = AverageMeter()
    loss_distribution_align_meter = AverageMeter()
    loss_cap_img_meter = AverageMeter()  
    loss_pair_meter = AverageMeter()  

    acc_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    num_classes = sum(args_train.classes)
    batch = args_train.batch_size
    i_ter = num_classes // batch + (1 if num_classes % batch else 0)

    epoch_total_loss = []
    epoch_reid_loss = []
    epoch_part_contrastive_loss = []
    epoch_distribution_align_loss = []
    epoch_cap_img_loss = []
    epoch_pair_loss = []

    # centers initialization
    print("initialize the centers")
    model.train()
    for i, (img, vid, _, _, domain, cid, captions, path) in enumerate(
        train_loader_stage2
    ):
        with torch.no_grad():
            img, target = img.to(device), vid.to(device)
            domain, cid = domain.to(device), cid.to(device)
            img_feats = model(img, target, get_image=True)  
            cap_feats = model(
                captions=captions, get_captions=True
            )  
            patch_centers.get_soft_label(
                path, img_feats, cap_feats, domain,vid=vid, camid=cid
            )
    print("initialization done")

    print("prepare ID prompt")
    text_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(i_ter):
            l_list = torch.arange(i * batch, min((i + 1) * batch, num_classes)).to(device)
            text_feat, _ = model(label=l_list, get_text=True)
            text_feats.append(text_feat)  
    prompt_feats = torch.cat(text_feats, 0).to(device)  
    print("prepare done")

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total_meter.reset()
        loss_reid_meter.reset()
        loss_cap_img_meter.reset()
        loss_pair_meter.reset()
        loss_partcontrastive_meter.reset()
        loss_distribution_align_meter.reset()

        acc_meter.reset()
        scheduler_image.step()

        for n_iter, (img, vid, _, _, domain, cid, captions, path) in enumerate(
            train_loader_stage2
        ):
            optimizer_image.zero_grad()
            img, target = img.to(device), vid.to(device)
            domain, cid = domain.to(device), cid.to(device)
            model.eval()
            with torch.no_grad():
                text_p, _ = model(
                    label=target, get_text=True, domain=domain, getdomain=False
                )  
            model.train()
            with amp.autocast(enabled=True):
                score, feat, global_feat = model(
                    x=img, label=target
                )  
                img_feats = model(
                    img, target, get_image=True
                ) 
                cap_feats = model(
                    captions=captions, get_captions=True
                )  
                global_img_feat = img_feats[3]  
                local_img_feats = torch.stack(
                    img_feats[:3], dim=0
                )  

                global_text_feat = cap_feats[3]  
                local_text_feats = torch.stack(
                    cap_feats[:3], dim=0
                )  

                global_img_feat = F.normalize(global_img_feat, dim=1)
                global_text_feat = F.normalize(global_text_feat, dim=1)

                patch_agent, patch_text_agent, position = patch_centers.get_soft_label(
                    path, img_feats, cap_feats, domain,vid=vid, camid=cid
                )
                loss_output = pc_criterion(
                    global_img_feat,
                    global_text_feat,
                    local_img_feats,
                    local_text_feats,
                    patch_agent,
                    patch_text_agent,
                    position,
                    patch_centers,
                    vid=target,
                    camid=cid,
                    domains=domain,
                )
                part_contrastive_loss = loss_output["part_contrastive_loss"]
                distribution_align_loss = loss_output["align_loss"]
                all_posvid = loss_output["all_posvid"]

                reid_loss = loss_fn[0](
                    score[0],
                    global_feat,
                    target,
                    all_posvid=all_posvid,
                    soft_label=True,
                    soft_weight=0.5,
                    soft_lambda=0.5,
                )

                loss_cap_img = 0.0
                for p in range(4):
                    norm_img_feat = F.normalize(img_feats[p].float(), p=2, dim=1)
                    norm_cap_feat = F.normalize(cap_feats[p].float(), p=2, dim=1)
                    loss_cap_img += clip_contrastive_loss(norm_img_feat, norm_cap_feat)
                loss_cap_img /= 4  

                logits = global_feat @ prompt_feats.t()
                loss_pair = F.cross_entropy(logits, target)

                total_loss = (
                    2.5 * reid_loss +
                    1.5 * loss_cap_img +
                    0.8 * loss_pair +
                    0.8 * part_contrastive_loss + # 1.2 -> 0.8
                    1.0 * distribution_align_loss # 1.5 -> 1.0
                )
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_image)
            scaler.update()

            acc = ((score[0] + score[1]).max(1)[1] == target).float().mean()
            batch_size = img.shape[0]
            loss_total_meter.update(total_loss.item(), batch_size)
            loss_reid_meter.update(reid_loss.item(), batch_size)
            loss_cap_img_meter.update(loss_cap_img.item(), batch_size)
            loss_pair_meter.update(loss_pair.item(), batch_size)
            loss_partcontrastive_meter.update(part_contrastive_loss.item(), batch_size)
            loss_distribution_align_meter.update(
                distribution_align_loss.item(), batch_size
            )

            acc_meter.update(acc, 1)
            torch.cuda.synchronize()


            if (n_iter + 1) % args_train.log_period == 0:
                acc = ((score[0] + score[1]).max(1)[1] == target).float().mean()
                logger_train.info(
                    f"STAGE2-Epoch[{epoch}] Iter[{n_iter+1}/{len(train_loader_stage2)}]\n"
                    f"  Total Loss: {loss_total_meter.avg:.4f} | "
                    f"ReID Loss: {loss_reid_meter.avg:.4f} | "
                    f"  Text-Img Loss: {loss_cap_img_meter.avg:.4f} | "
                    f"Pair Loss: {loss_pair_meter.avg:.4f}\n"
                    f"  Part Contrastive Loss: {loss_partcontrastive_meter.avg:.4f} | "
                    f"Distribution Align Loss: {loss_distribution_align_meter.avg:.4f}\n"
                    f"  Acc: {acc:.3f} | Lr: {scheduler_image.get_lr()[0]:.2e}"
                )
        epoch_total_loss.append(loss_total_meter.avg)
        epoch_reid_loss.append(loss_reid_meter.avg)
        epoch_part_contrastive_loss.append(loss_partcontrastive_meter.avg)
        epoch_distribution_align_loss.append(loss_distribution_align_meter.avg)
        epoch_cap_img_loss.append(loss_cap_img_meter.avg)
        epoch_pair_loss.append(loss_pair_meter.avg)

        if epoch % args_train.checkpoint_period == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    log_path,
                    f"{args_train.model}_{datetime.datetime.now()}_stage2_{epoch}.pth",
                ),
            )
        if epoch % args_train.eval_period == 0:
            test(testloaders, model, logger_test)

def test(testloaders, model, logger_test):
    model.eval()
    maps, r1s = [], []
    for name, val_loader in testloaders.items():
        evaluator = R1_mAP_eval(val_loader[1], max_rank=10, feat_norm=True)
        evaluator.reset()
        logger_test.info(f"Validation Results of {name}: ")
        for _, (img, pids, camids, _, _, _, _, _) in enumerate(val_loader[0]):
            with torch.no_grad():
                feat = model(img.cuda())
                evaluator.update((feat, pids, camids))
        cmc, mAP = evaluator.compute()[:2]
        logger_test.info(
            f"mAP: {mAP:.1%} | Rank-1: {cmc[0]:.1%}, Rank-5: {cmc[4]:.1%}, Rank-10: {cmc[9]:.1%}"
        )
        logger_test.info("-" * 30)
        maps.append(mAP), r1s.append(cmc[0])

    logger_test.info(
        f"Average mAP: {sum(maps)/len(maps):.1%} | Average Rank-1: {sum(r1s)/len(r1s):.1%}"
    )


def clip_contrastive_loss(image_features, text_features, temperature=0.1):
    logits_per_image = torch.matmul(image_features, text_features.t()) / temperature
    logits_per_text = logits_per_image.t()

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size).to(image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


if __name__ == "__main__":
    seed = 1234 
    set_seed(seed)
    parser, parser_test = argparse.ArgumentParser(
        description="train"
    ), argparse.ArgumentParser(description="test")

    # todo
    parsertrain, parsertest, logname =  protocol_1(parser, parser_test)

    # parsertrain, parsertest, logname = protocol_2_MS(parser, parser_test)
    # parsertrain, parsertest, logname = protocol_2_C3(parser, parser_test)
    # parsertrain, parsertest, logname = protocol_2_M(parser, parser_test)
  
    # parsertrain, parsertest, logname = protocol_3_MS(parser, parser_test)
    # parsertrain, parsertest, logname = protocol_3_C3(parser, parser_test)
    # parsertrain, parsertest, logname = protocol_3_M(parser, parser_test)

    args_train, args_test = parsertrain.parse_args(), parsertest.parse_args()

    log_suffix = args_train.log_suffix if args_train.log_suffix else f"test"
    log_path = os.path.join(
        args_train.log_path,
        log_suffix
    )
    
    logger_train = setup_logger(
        f"{args_train.model}_{args_train.backbone}_train", log_path, if_train=True
    )
    logger_test = setup_logger(
        f"{args_train.model}_{args_train.backbone}_test", log_path, if_train=False
    )
    logger_train.info(
        f"Log saved in {log_path} | Protocol: {args_train.train_datasets}->{args_test.test_datasets}"
    )

    model = get_model(args_train)
    train_loader_stage1, train_loader_stage2, val_loaders = build_data_loader(
        args_train, args_test
    )
    criterion = make_loss(sum(args_train.classes))

    patch_centers = Patchloss.PatchMemory(momentum=0.3, num=1)
    pc_criterion = Patchloss.Pedal(
        scale=0.02,  
        k=10,
        kernel_bandwidth=0.65 
    ).cuda()
    
    num_classes = len(train_loader_stage2.dataset.pids)
    loss_func = build_loss(num_classes=num_classes)

    # init
    optimizer_init = make_optimizer(model, args_train, 'init')
    scheduler_init = WarmupMultiStepLR(
        optimizer_init, [30, 50], 0.1, 0.1, 10, "linear"
    )

    init_image_encoder(
        train_loader_stage2,
        model,
        criterion,
        optimizer_init,
        scheduler_init,
        args_train,
        logger_train,
        log_path,
        args_train.prior_img_epoch,
    )

    optimizer_stage1 = make_optimizer(model, args_train,'stage1')
    scheduler_stage1 = create_scheduler(
        args_train.prompt_epoch, args_train.prompt_lr, optimizer_stage1
    )

    train_stage1(
        train_loader_stage1,
        model,
        optimizer_stage1,
        scheduler_stage1,
        args_train,
        logger_train,
        log_path,
        get_domain=False,
        epochs=args_train.prompt_epoch,
    )

    optimizer_stage2 = make_optimizer(model, args_train, 'stage2')
    scheduler_stage2 = WarmupMultiStepLR(
        optimizer_stage2, [40, 55], 0.1, 0.1, 10, "linear"
    )

    train_stage2(
        train_loader_stage2,
        model,
        criterion,
        optimizer_stage2,
        scheduler_stage2,
        val_loaders,
        args_train,
        logger_train,
        logger_test,
        log_path,
        epochs=args_train.image_encoder_epoch,
        loss_fn=loss_func,
    )

    test(val_loaders, model, logger_test)
