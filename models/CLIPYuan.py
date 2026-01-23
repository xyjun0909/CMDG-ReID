import torch
import torch.nn as nn
import numpy as np
from .clip import clip
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class TextEncoderForCaptions(nn.Module):
    def __init__(self, clip_model, fine_tune_layers=4):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.fine_tune_layers = fine_tune_layers
        self.token_embedding = clip_model.token_embedding

        total_layers = len(self.transformer.resblocks)
        for i, block in enumerate(self.transformer.resblocks):
            if i < (total_layers - self.fine_tune_layers):
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


# 原有域分类器（保留）
class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        out = self.fc1(x)
        if x.shape[0] == 1:
            out = out.repeat(2, 1)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return F.log_softmax(out, dim=1)[0]
        else:
            out = self.bn1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return F.log_softmax(out, dim=1)


# 原有梯度反转层（保留）
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None


class GRL(torch.nn.Module):
    def __init__(self, lambd=.1):
        super(GRL, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


# 新增BAU域分类器
class BAUDomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        # BAU特有的初始化方式
        self.fc1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


# 新增BAU梯度反转层
class BAUGradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None


class BAUGRL(torch.nn.Module):
    def __init__(self, lambd=0.2):  # BAU默认lambda值不同
        super(BAUGRL, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return BAUGradReverse.apply(x, lam)


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_num, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        ctx_init_domain = "A photo of a X X X X person from X dataset."

        ctx_dim = 512
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        tokenized_prompts_domain = clip.tokenize(ctx_init_domain).cuda()

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding_domain = token_embedding(tokenized_prompts_domain).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_domain = tokenized_prompts_domain

        n_cls_ctx = 4
        n_dm_ctx = 1
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        dom_vectors = torch.empty(dataset_num, n_dm_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(dom_vectors, std=0.02)
        self.clsctx = nn.Parameter(cls_vectors, requires_grad=True)
        self.dmctx = nn.Parameter(dom_vectors, requires_grad=True)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])

        self.register_buffer("token_prefix_domain", embedding_domain[:, :n_ctx + 1, :])
        self.register_buffer("token_intermediate_domain", embedding_domain[:, n_ctx + 1 + n_cls_ctx:n_ctx + 1 + n_cls_ctx+2, :])
        self.register_buffer("token_suffix_domain", embedding_domain[:, n_ctx + 1 + n_cls_ctx+2 + n_dm_ctx:, :])

        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label, domain=None):
        if domain is not None:
            cls_ctx = self.clsctx[label]
            cls_ctx_clone = cls_ctx.clone().detach()
            b = label.shape[0]
            dom_ctx = self.dmctx[domain]
            prefix = self.token_prefix_domain.expand(b, -1, -1)
            intermediate = self.token_intermediate_domain.expand(b, -1, -1)
            suffix = self.token_suffix_domain.expand(b, -1, -1)
            prompts = torch.cat(
                [prefix, cls_ctx_clone, intermediate, dom_ctx, suffix], dim=1
            )
            return prompts

        cls_ctx = self.clsctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)
        return prompts


class Model(nn.Module):
    def __init__(self, num_classes, args, epsilon=.1, domain_num=4):
        super(Model, self).__init__()
        self.h_resolution = int((args.size_train[0] - 16) // 16 + 1)
        self.w_resolution = int((args.size_train[1] - 16) // 16 + 1)
        self.vision_stride_size = 16
        self.model_name = args.backbone
        self.neck_feat = 'before'
        self.num_regions = 3

        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'ViT-B-32':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 32) // 32 + 1)
            self.w_resolution = int((args.size_train[1] - 32) // 32 + 1)
            self.vision_stride_size = 32
        elif self.model_name == 'ViT-L-14':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 14) // 14 + 1)
            self.w_resolution = int((args.size_train[1] - 14) // 14 + 1)
            self.vision_stride_size = 14
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        elif self.model_name == 'RN101':
            self.in_planes = 2048
            self.in_planes_proj = 512

        self.img_norm = nn.LayerNorm(self.in_planes)
        self.img_norm_proj = nn.LayerNorm(self.in_planes_proj)

        self.num_classes = num_classes
        # 保留原有梯度反转层
        self.grl = GRL(epsilon)
        # 新增BAU梯度反转层
        self.bau_grl = BAUGRL(lambd=0.2)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        
        # 保留原有域分类器
        self.dcgrl = DomainClassifier(self.in_planes_proj, 128, domain_num)
        self.dc = DomainClassifier(self.in_planes_proj, 128, domain_num)
        # 新增BAU域分类器
        self.bau_dcgrl = BAUDomainClassifier(self.in_planes_proj, 256, domain_num)  # BAU隐藏层维度不同
        self.bau_dc = BAUDomainClassifier(self.in_planes_proj, 256, domain_num)

        self.image_encoder = clip_model.visual
        self.promptlearner = PromptLearner(num_classes, domain_num, clip_model.dtype, clip_model.token_embedding)
        self.caption_tokenizer = clip.tokenize
        self.text_encoder = TextEncoderForCaptions(clip_model)
        self.local_proj = nn.Linear(self.in_planes, self.in_planes)
        self.local_proj.apply(weights_init_kaiming)
        self.local_proj_proj = nn.Linear(self.in_planes_proj, self.in_planes_proj)
        self.local_proj_proj.apply(weights_init_kaiming)

    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None,
                domain=None, prior=False, getdomain=False, captions=None, get_captions=False, use_bau=False):
        if get_captions:
            return self.encode_captions(captions)

        if get_text:
            if getdomain:
                prompts = self.promptlearner(label, domain)
                text_features = self.text_encoder(prompts, self.promptlearner.tokenized_prompts_domain)
                # 根据use_bau参数选择分类器
                if use_bau:
                    resd = self.bau_dc(text_features.clone().detach())
                else:
                    resd = self.dc(text_features.clone().detach())
                return text_features, resd

            prompts = self.promptlearner(label)
            text_features = self.text_encoder(prompts, self.promptlearner.tokenized_prompts)
            if use_bau:
                resd = self.bau_dcgrl(self.bau_grl(text_features))
            else:
                resd = self.dcgrl(self.grl(text_features))
            return text_features, resd

        if get_image:
            cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            region_features, region_features_proj = self.get_image_features(x, image_features, image_features_proj)
            return region_features_proj

        cv_embed = None
        image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
        img_feature_last = image_features_last[:, 0]
        region_features, region_features_proj = self.get_image_features(x, image_features, image_features_proj)
        global_feature = region_features[-1]
        global_feature_proj = region_features_proj[-1]
        feat = self.bottleneck(global_feature)
        feat_proj = self.bottleneck_proj(global_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, global_feature, global_feature_proj], global_feature_proj
        else:
            return torch.cat([feat, feat_proj], dim=1) if self.neck_feat == 'after' else torch.cat([global_feature, global_feature_proj], dim=1)

    def encode_captions(self, captions):
        batch_size = len(captions)
        caption_features = []
        for i in [1, 2, 3, 0]:
            texts = [captions[b][i] for b in range(batch_size)]
            tokenized = self.caption_tokenizer(texts).cuda()
            embeddings = self.text_encoder.token_embedding(tokenized).type(self.text_encoder.dtype)
            features = self.text_encoder(embeddings, tokenized)
            caption_features.append(features)
        return caption_features

    def get_image_features(self, x, image_features, image_features_proj):
        global_feature = image_features[:, 0]
        global_feature_proj = image_features_proj[:, 0]
        
        patch_features = image_features[:, 1:].reshape(
            x.shape[0], self.h_resolution, self.w_resolution, -1
        ).permute(0, 3, 1, 2)
        patch_features_proj = image_features_proj[:, 1:].reshape(
            x.shape[0], self.h_resolution, self.w_resolution, -1
        ).permute(0, 3, 1, 2)

        region_features, region_features_proj = [], []
        region_height = self.h_resolution // self.num_regions
        for i in range(self.num_regions):
            start_h = i * region_height
            end_h = (i + 1) * region_height if i < 2 else self.h_resolution
            region = F.adaptive_avg_pool2d(patch_features[:, :, start_h:end_h, :], (1, 1)).squeeze(-1).squeeze(-1)
            region_proj = F.adaptive_avg_pool2d(patch_features_proj[:, :, start_h:end_h, :], (1, 1)).squeeze(-1).squeeze(-1)
            region_features.append(region)
            region_features_proj.append(region_proj)

        region_features.append(global_feature)
        region_features_proj.append(global_feature_proj)
        
        region_features = [self.img_norm(f) for f in region_features]
        region_features_proj = [self.img_norm_proj(f) for f in region_features_proj]
        
        return region_features, region_features_proj

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))