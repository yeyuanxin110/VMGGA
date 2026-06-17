import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Union, Tuple
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ResNetFPN_8_2(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Dinov3 Backbone
        self.dinov3 = Dinov3()
        embed_dim = self.dinov3.model.embed_dim
        self.fuse_convs = nn.ModuleList([
            nn.Conv2d(embed_dim + block_dims[0], block_dims[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + block_dims[0], block_dims[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + block_dims[1], block_dims[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + block_dims[2], block_dims[2], kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.fuse_norms = nn.ModuleList([
            nn.SyncBatchNorm(block_dims[0]),
            nn.SyncBatchNorm(block_dims[0]),
            nn.SyncBatchNorm(block_dims[1]),
            nn.SyncBatchNorm(block_dims[2])
        ])
    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # Dinov3 Backbone
        X_dino = self.dinov3(x)  # 4 layers at 1/16 scale
        target_sizes = [None, x1.shape[2:], x2.shape[2:], x3.shape[2:]]
        X_dino_resample = []
        for i, x_dino in enumerate(X_dino):
            if i == 0:
                X_dino_resample.append(None)
            else:
                X_dino_resample.append(F.interpolate(x_dino, size=target_sizes[i],
                                       mode='bilinear', align_corners=True))

        # Fusion
        # c0 = torch.cat([X_dino_resample[0], x0], dim=1)
        c1 = torch.cat([X_dino_resample[1], x1], dim=1)
        c2 = torch.cat([X_dino_resample[2], x2], dim=1)
        c3 = torch.cat([X_dino_resample[3], x3], dim=1)
        # x0 = self.fuse_convs[0](c0)
        x1 = self.fuse_convs[1](c1)
        x2 = self.fuse_convs[2](c2)
        x3 = self.fuse_convs[3](c3)

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return [x3_out, x1_out]  # x3_out: 2*256*80*80, x1_out: 2*128*320*320

class ResNetFPN_8_2_without_dinov3(nn.Module):
    """
    ResNet+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        return [x3_out, x1_out]

class ConcatFusion(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        # 拼接后通道数为256+256=512，用1×1卷积压缩至256
        self.conv_compress = nn.Conv2d(
            in_channels * 2,  # 输入通道：两个特征的通道和
            out_channels,     # 输出通道：目标通道256
            kernel_size=1,    # 仅压缩通道，不改变空间尺寸
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()  # 激活函数：增加非线性

    def forward(self, feat1, feat2):
        # 通道拼接（dim=1：通道维度）
        concat_feat = torch.cat([feat1, feat2], dim=1)  # 输出：(1, 512, 80, 80)
        # 压缩通道并激活
        fused_feat = self.relu(self.conv_compress(concat_feat))  # 输出：(1,256,80,80)
        return fused_feat

class OneStepConverter(nn.Module):
    def __init__(self):
        super().__init__()
        # 转置卷积：同时压缩通道（768→256）和放大空间（40×40→80×80）
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=768,
            out_channels=256,
            kernel_size=4,  # 3×3卷积核，平衡感受野和计算量
            stride=2,  # 步长=2，输出尺寸变为输入的2倍（40×2=80）
            padding=1,  # 配合stride=2，保证输出尺寸准确
            output_padding=0  # 额外补充0，确保40×2=80（无多余像素）
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 输入：(B, 768, 40, 40) → 输出：(B, 256, 80, 80)
        x = self.transpose_conv(x)
        x = self.relu(x)
        return x

class Dinov3(nn.Module):
    def __init__(self, model_name='dinov3_vitb16'):
        super().__init__()
        self.model = None
        model_feature_layers = {
            'dinov3_vits16': [3, 5, 7, 11],
            'dinov3_vits16plus': [3, 5, 7, 11],
            'dinov3_vitb16': [3, 5, 7, 11],
            'dinov3_vitl16': [7, 11, 15, 23],
            'dinov3_vith16plus': [9, 13, 18, 26],
            'dinov3_vit7b16': [11, 16, 21, 31]
        }
        self.interaction_indexes = model_feature_layers.get(model_name)
        if model_name == 'dinov3_vitb16':
            import sys, os
            _dinov3_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dinov3')
            if _dinov3_root not in sys.path:
                sys.path.insert(0, _dinov3_root)
            from .dinov3.dinov3.hub.backbones import dinov3_vitb16
            self.model = dinov3_vitb16(pretrained=False)
        self.n_layers = self.model.n_blocks
        self.patch_size = self.model.patch_size


    def get_Dinov3_patchtokens(self, im, reshape):
        im_size = im.size()[2:]
        # x_ = self.model(im, is_training=True)
        x = self.model(im, is_training=True)['x_norm_patchtokens']
        if reshape:
            x = self.patch_token_to_featmap(x, self.patch_size, im_size)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def get_Dinov3_last_N_layer(self, im, reshape):
        x = self.model.get_intermediate_layers(im, n=self.interaction_indexes, reshape=reshape, norm=True)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def patch_token_to_featmap(self,
            patch_tokens: torch.Tensor,
            patch_size: int,
            img_size: Union[Tuple[int, int], int]
    ) -> torch.Tensor:
        """
        将patch token自动重塑为特征图，仅需输入patch token、patch_size、原图大小。

        参数:
            patch_tokens: 输入的patch token张量，形状为 (batch_size, num_patches, embed_dim)
            patch_size: 图像分割时的patch大小（如16、8）
            img_size: 原图尺寸，整数（正方形）或元组（高, 宽），如224或(224, 224)

        返回:
            特征图张量，形状为 (batch_size, embed_dim, feat_h, feat_w)
        """
        # 1. 自动提取已知维度（batch_size、num_patches、embed_dim）
        batch_size = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        embed_dim = patch_tokens.shape[2]  # 自动获取通道维度，无需手动输入

        # 2. 处理原图尺寸（统一转为元组格式）
        if isinstance(img_size, int):
            img_h = img_w = img_size
        else:
            img_h, img_w = img_size

        # 3. 自动计算特征图的空间维度（feat_h/feat_w），并校验尺寸匹配
        feat_h = img_h // patch_size  # 特征图高度 = 原图高度 ÷ patch_size
        feat_w = img_w // patch_size  # 特征图宽度 = 原图宽度 ÷ patch_size
        # 校验：确保patch数量与空间维度匹配（避免原图尺寸不能被patch_size整除）
        assert feat_h * feat_w == num_patches, \
            f"patch数量不匹配！原图{img_h}×{img_w}、patch_size={patch_size} → 应生成{feat_h * feat_w}个patch，但输入token有{num_patches}个"

        # 4. 重塑为特征图（核心步骤）
        # 第一步：将num_patches拆分为空间维度 (batch_size, feat_h, feat_w, embed_dim)
        reshaped = patch_tokens.view(batch_size, feat_h, feat_w, embed_dim)
        # 第二步：调整维度顺序为 (batch_size, embed_dim, feat_h, feat_w)（视觉任务标准格式）
        featmap = reshaped.permute(0, 3, 1, 2).contiguous()

        return featmap

    def forward(self, x):
        # x4 = self.get_Dinov3_patchtokens(x.repeat(1, 3, 1, 1), reshape=False) # 1/16
        x4 = self.get_Dinov3_last_N_layer(x.repeat(1, 3, 1, 1), reshape=True)  # 1/16
        return x4


class ResNetFPN_16_4(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4)

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        return [x4_out, x2_out]


class Dinov3Only_FPN_8_2(nn.Module):
    """
    DINOv3 + FPN (Without ResNet Backbone)
    Output resolution are 1/8 and 1/2.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # 1. DINOv3 Backbone (完全替代 ResNet)
        self.dinov3 = Dinov3()
        embed_dim = self.dinov3.model.embed_dim

        # 2. 投影层 (Projection Layers)
        # 因为没有了 ResNet 特征做 Concat，我们只需将 DINOv3 的 embed_dim (768)
        # 降维到 FPN 期望的 block_dims: [1/2尺度, 1/4尺度, 1/8尺度]
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, block_dims[0], kernel_size=1, stride=1, padding=0, bias=False),  # 对应 1/2 (x1)
            nn.Conv2d(embed_dim, block_dims[1], kernel_size=1, stride=1, padding=0, bias=False),  # 对应 1/4 (x2)
            nn.Conv2d(embed_dim, block_dims[2], kernel_size=1, stride=1, padding=0, bias=False)  # 对应 1/8 (x3)
        ])

        # 对应投影后的 BatchNorm
        self.proj_norms = nn.ModuleList([
            nn.SyncBatchNorm(block_dims[0]),
            nn.SyncBatchNorm(block_dims[1]),
            nn.SyncBatchNorm(block_dims[2])
        ])

        # 3. FPN 结构 (与原版完全保持一致，保证消融实验的公平性)
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        # 初始化 FPN 和投影层的参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. DINOv3 特征提取 (输入 x 会在内部被复制为 3 通道)
        X_dino = self.dinov3(x)  # 得到4个层的特征，原始分辨率均为 1/16

        # 2. 对 DINOv3 特征进行上采样插值，对齐 FPN 需要的尺度
        X_dino_resample = []
        for i, x_dino in enumerate(X_dino):
            if i == 0:
                X_dino_resample.append(None)  # 第一个特征不用
            else:
                # i=1 -> scale_factor=8 (1/2尺度)
                # i=2 -> scale_factor=4 (1/4尺度)
                # i=3 -> scale_factor=2 (1/8尺度)
                X_dino_resample.append(
                    F.interpolate(x_dino, scale_factor=2 ** (4 - i), mode='bilinear', align_corners=True))

        # 3. 降维投影 (替代原先的 Fusion)
        # 直接把插值后的 DINOv3 变成原来 ResNet 输出的 x1, x2, x3 的形状
        x1 = self.proj_norms[0](self.proj_convs[0](X_dino_resample[1]))
        x2 = self.proj_norms[1](self.proj_convs[1](X_dino_resample[2]))
        x3 = self.proj_norms[2](self.proj_convs[2](X_dino_resample[3]))

        # 4. FPN (特征金字塔)
        x3_out = self.layer3_outconv(x3)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out + x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        return [x3_out, x1_out]
