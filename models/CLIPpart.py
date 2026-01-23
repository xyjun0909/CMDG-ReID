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
        # loading JIT archive
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

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


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


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None

# 梯度反转层
class GRL(torch.nn.Module):
    def __init__(self, lambd=.1):
        super(GRL, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_num, dtype, token_embedding, num_regions=3):
        super().__init__()
        # 提示词模板
        global_ctx_init = "A photo of a X X X X person."
        global_ctx_init_domain = "A photo of a X X X X person from X dataset."
        
        # 局部提示词模板（上、中、下）
        region_ctx_inits = [
            "The upper part of a photo of a X X X X person.",
            "The middle part of a photo of a X X X X person.",
            "The lower part of a photo of a X X X X person."
        ]
        
        region_ctx_inits_domain = [
            "The upper part of a photo of a X X X X person from X dataset.",
            "The middle part of a photo of a X X X X person from X dataset.",
            "The lower part of a photo of a X X X X person from X dataset."
        ]
        
        ctx_dim = 512
        n_ctx = 4  # 每个提示词的可学习上下文向量数量
        
        # 初始化tokenized提示词
        tokenized_prompts = clip.tokenize(global_ctx_init).cuda()
        tokenized_prompts_domain = clip.tokenize(global_ctx_init_domain).cuda()
        tokenized_region_prompts = [clip.tokenize(p).cuda() for p in region_ctx_inits]
        tokenized_region_prompts_domain = [clip.tokenize(p).cuda() for p in region_ctx_inits_domain]
        
        # 获取提示词嵌入
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
            embedding_domain = token_embedding(tokenized_prompts_domain).type(dtype)
            region_embeddings = [token_embedding(p).type(dtype) for p in tokenized_region_prompts]
            region_embeddings_domain = [token_embedding(p).type(dtype) for p in tokenized_region_prompts_domain]
        
        # 保存tokenized提示词
        self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_domain = tokenized_prompts_domain
        self.tokenized_region_prompts = tokenized_region_prompts
        self.tokenized_region_prompts_domain = tokenized_region_prompts_domain
        
        # 全局可学习上下文
        n_cls_ctx = 4
        n_dm_ctx = 1
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        dom_vectors = torch.empty(dataset_num, n_dm_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        nn.init.normal_(dom_vectors, std=0.02)
        self.clsctx = nn.Parameter(cls_vectors, requires_grad=True)  # 全局类别上下文
        self.dmctx = nn.Parameter(dom_vectors, requires_grad=True)   # 全局领域上下文
        
        # 局部可学习上下文（每个区域独立）
        self.local_clsctx = nn.ParameterList([
            nn.Parameter(torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype), requires_grad=True)
            for _ in range(num_regions)
        ])
        self.local_dmctx = nn.ParameterList([
            nn.Parameter(torch.empty(dataset_num, n_dm_ctx, ctx_dim, dtype=dtype), requires_grad=True)
            for _ in range(num_regions)
        ])
        
        # 初始化局部上下文
        for i in range(num_regions):
            nn.init.normal_(self.local_clsctx[i], std=0.02)
            nn.init.normal_(self.local_dmctx[i], std=0.02)
        
        # 保存提示词前缀和后缀
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        
        self.register_buffer("token_prefix_domain", embedding_domain[:, :n_ctx + 1, :])
        self.register_buffer("token_intermediate_domain", embedding_domain[:, n_ctx + 1 + n_cls_ctx:n_ctx + 1 + n_cls_ctx+2, :])
        self.register_buffer("token_suffix_domain", embedding_domain[:, n_ctx + 1 + n_cls_ctx+2 + n_dm_ctx:, :])
        
        # 保存局部提示词的前缀和后缀
        self.token_region_prefixes = nn.ParameterList([
            nn.Parameter(region_embeddings[i][:, :n_ctx + 1, :], requires_grad=False)
            for i in range(num_regions)
        ])
        self.token_region_suffixes = nn.ParameterList([
            nn.Parameter(region_embeddings[i][:, n_ctx + 1 + n_cls_ctx:, :], requires_grad=False)
            for i in range(num_regions)
        ])
        
        self.token_region_prefixes_domain = nn.ParameterList([
            nn.Parameter(region_embeddings_domain[i][:, :n_ctx + 1, :], requires_grad=False)
            for i in range(num_regions)
        ])
        self.token_region_intermediates_domain = nn.ParameterList([
            nn.Parameter(region_embeddings_domain[i][:, n_ctx + 1 + n_cls_ctx:n_ctx + 1 + n_cls_ctx+2, :], requires_grad=False)
            for i in range(num_regions)
        ])
        self.token_region_suffixes_domain = nn.ParameterList([
            nn.Parameter(region_embeddings_domain[i][:, n_ctx + 1 + n_cls_ctx+2 + n_dm_ctx:, :], requires_grad=False)
            for i in range(num_regions)
        ])
        
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx
        self.num_regions = num_regions

    def forward(self, label, domain=None, region_idx=None):
        """
        生成提示词嵌入
        - region_idx=None: 生成全局提示词
        - region_idx=0/1/2: 生成对应区域的局部提示词
        """
        if region_idx is not None:  # 局部提示词
            assert 0 <= region_idx < self.num_regions, f"Invalid region index: {region_idx}"
            
            if domain is not None:  # 领域相关局部提示词
                cls_ctx = self.local_clsctx[region_idx][label]
                b = label.shape[0]
                dom_ctx = self.local_dmctx[region_idx][domain]
                
                prefix = self.token_region_prefixes_domain[region_idx].expand(b, -1, -1)
                intermediate = self.token_region_intermediates_domain[region_idx].expand(b, -1, -1)
                suffix = self.token_region_suffixes_domain[region_idx].expand(b, -1, -1)
                
                prompts = torch.cat(
                    [
                        prefix,  # (b, 1, dim)
                        cls_ctx,  # (b, n_ctx, dim)
                        intermediate,  # (b, *, dim)
                        dom_ctx,
                        suffix,
                    ],
                    dim=1,
                )
                return prompts
            
            # 领域无关局部提示词
            cls_ctx = self.local_clsctx[region_idx][label]
            b = label.shape[0]
            
            prefix = self.token_region_prefixes[region_idx].expand(b, -1, -1)
            suffix = self.token_region_suffixes[region_idx].expand(b, -1, -1)
            
            prompts = torch.cat(
                [
                    prefix,  # (b, 1, dim)
                    cls_ctx,  # (b, n_ctx, dim)
                    suffix,  # (b, *, dim)
                ],
                dim=1,
            )
            return prompts
        
        # 全局提示词
        if domain is not None:  # 领域相关全局提示词
            cls_ctx = self.clsctx[label]
            b = label.shape[0]
            dom_ctx = self.dmctx[domain]
            
            prefix = self.token_prefix_domain.expand(b, -1, -1)
            intermediate = self.token_intermediate_domain.expand(b, -1, -1)
            suffix = self.token_suffix_domain.expand(b, -1, -1)
            
            prompts = torch.cat(
                [
                    prefix,  # (b, 1, dim)
                    cls_ctx,  # (b, n_ctx, dim)
                    intermediate,  # (b, *, dim)
                    dom_ctx,
                    suffix,
                ],
                dim=1,
            )
            return prompts
        
        # 领域无关全局提示词
        cls_ctx = self.clsctx[label]
        b = label.shape[0]
        
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        
        prompts = torch.cat(
            [
                prefix,  # (b, 1, dim)
                cls_ctx,  # (b, n_ctx, dim)
                suffix,  # (b, *, dim)
            ],
            dim=1,
        )
        return prompts


class Model(nn.Module):
    def __init__(self, num_classes, args, epsilon=.1, domain_num=4):
        super(Model, self).__init__()
        self.h_resolution = int((args.size_train[0] - 16) // 16 + 1)
        self.w_resolution = int((args.size_train[1] - 16) // 16 + 1)
        self.vision_stride_size = 16
        self.model_name = args.backbone
        self.neck_feat = 'before'
        
        # 设置输入特征维度
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
        
        self.num_classes = num_classes
        self.grl = GRL(epsilon)
        self.num_regions = 3  # 上中下三个区域
        
        # 分类器
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        
        # 瓶颈层
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        
        # 加载CLIP模型
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        
        # 领域分类器
        self.dcgrl = DomainClassifier(self.in_planes_proj, 128, domain_num)
        self.dc = DomainClassifier(self.in_planes_proj, 128, domain_num)
        
        # 图像编码器
        self.image_encoder = clip_model.visual
        
        # 提示词学习器（支持局部和全局）
        self.promptlearner = PromptLearner(num_classes, domain_num, clip_model.dtype, clip_model.token_embedding, self.num_regions)
        
        # 文本编码器
        self.text_encoder = TextEncoder(clip_model)
        
         # 添加层归一化模块
        self.img_norm = nn.LayerNorm(self.in_planes)
        self.text_norm = nn.LayerNorm(self.in_planes)

    def get_image_features(self, x):
        """提取图像的全局和局部特征（基于空间划分）原本是直接选定特定 tokens 的局部特征"""
        # 获取ViT的输出
        if "RN" in self.model_name:
            # 对于ResNet模型，使用平均池化模拟区域划分
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]
            
            # 模拟区域特征（这里简化处理，实际应用中可能需要更复杂的分割）
            region_features = [img_feature for _ in range(self.num_regions)]
            region_features_proj = [img_feature_proj for _ in range(self.num_regions)]
            
        elif "ViT" in self.model_name:
            # 对于ViT模型，提取cls_token和patch tokens
            cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            
            # 全局特征：cls_token
            global_feature = image_features[:, 0]
            global_feature_proj = image_features_proj[:, 0]
            
            # 局部特征：将patch tokens按空间位置划分为三个区域
            patch_features = image_features[:, 1:]  # [B, num_patches, dim]
            patch_features_proj = image_features_proj[:, 1:]  # [B, num_patches, dim]
            
            # 重塑为空间布局 [B, h_res, w_res, dim]
            patch_features = patch_features.reshape(
                x.shape[0], self.h_resolution, self.w_resolution, -1
            ).permute(0, 3, 1, 2)  # [B, dim, h_res, w_res]
            
            patch_features_proj = patch_features_proj.reshape(
                x.shape[0], self.h_resolution, self.w_resolution, -1
            ).permute(0, 3, 1, 2)  # [B, dim, h_res, w_res]
            
            # 划分区域（上中下）
            region_features = []
            region_features_proj = []
            
            # 计算每个区域的高度
            region_height = self.h_resolution // self.num_regions
            
            for i in range(self.num_regions):
                start_h = i * region_height
                end_h = (i + 1) * region_height if i < self.num_regions - 1 else self.h_resolution
                
                # 提取区域特征并池化
                region = patch_features[:, :, start_h:end_h, :]
                region_proj = patch_features_proj[:, :, start_h:end_h, :]
                
                # 对区域进行全局平均池化
                region_pooled = F.adaptive_avg_pool2d(region, (1, 1)).squeeze(-1).squeeze(-1)
                region_proj_pooled = F.adaptive_avg_pool2d(region_proj, (1, 1)).squeeze(-1).squeeze(-1)
                
                region_features.append(region_pooled)
                region_features_proj.append(region_proj_pooled)
            
            # 添加全局特征
            region_features.append(global_feature)
            region_features_proj.append(global_feature_proj)
        
        # 应用层归一化
        region_features = [self.img_norm(feat) for feat in region_features]
        region_features_proj = [self.img_norm(feat) for feat in region_features_proj]
        
        return region_features, region_features_proj
    
    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None,
                domain=None, prior=False, getdomain=False, region_idx=None):
        if get_text:
            if getdomain:
                # 生成领域相关提示词并获取文本特征
                prompts = self.promptlearner(label, domain, region_idx)
                text_features = self.text_encoder(prompts, self.tokenized_prompts_domain)
                resd = self.dc(text_features.clone().detach())
                return text_features, resd

            # 生成领域无关提示词并获取文本特征
            prompts = self.promptlearner(label, None, region_idx)
            text_features = self.text_encoder(prompts, self.tokenized_prompts)
            resd = self.dcgrl(self.grl(text_features))
            return text_features, resd

        if get_image == True:
            # 只获取图像特征
            region_features, region_features_proj = self.get_image_features(x)
            if region_idx is not None:
                return region_features_proj[region_idx]
            return region_features_proj[-1]  # 默认返回全局特征

        # 训练模式：同时获取图像和文本特征
        region_features, region_features_proj = self.get_image_features(x)
        
        # 全局特征处理
        global_feature = region_features[-1]
        global_feature_proj = region_features_proj[-1]
        
        # 应用瓶颈层
        feat = self.bottleneck(global_feature)
        feat_proj = self.bottleneck_proj(global_feature_proj)
        
        if self.training:
            # 分类分数
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            
            # 返回全局和局部特征
            return [cls_score, cls_score_proj], [region_features, region_features_proj], global_feature_proj

        else:
            # 推理模式：返回特征
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([global_feature, global_feature_proj], dim=1)


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
