# Ultralytics YOLO, AGPL-3.0 license
"""DRG and C3k2 wrappers ported from the supplied mianju module pack.

DRPCA-Net, TGRS: https://arxiv.org/abs/2507.09541
This ports its Dynamic Residual Group, not the full RPCA unfolding network.
The pack's operators, defaults and eight-block layout are retained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .block import C3k, C3k2

__all__ = ("DRG", "C3k_DRG", "C3k2_DRG")

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class DynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, kernel_size**2, kernel_size=1)  # [B, k*k, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 每个样本生成一个动态卷积核 [B, k*k, 1, 1] → [B, 1, k, k]
        kernels = self.kernel_generator(x).view(B, 1, self.kernel_size, self.kernel_size)
        # 2. 对每个样本取通道平均 [B, 1, H, W]
        x_mean = x.mean(dim=1, keepdim=True)
        # 3. reshape 成 grouped convolution 所需格式
        x_mean = x_mean.view(1, B, H, W)  # → [1, B, H, W]
        kernels = kernels.view(B, 1, self.kernel_size, self.kernel_size)  # [B, 1, k, k]
        # 4. 执行 grouped convolution，每个 kernel 只作用于对应的样本
        att = F.conv2d(
            x_mean,
            weight=kernels,
            padding=self.kernel_size // 2,
            groups=B
        )
        # 5. reshape 回原格式 + sigmoid
        att = att.view(B, 1, H, W)
        att = self.sigmoid(att)
        # 6. 应用注意力图
        return x * att

# Residual Channel Attention Block (RCAB)
class RCSAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ChannelAttention(n_feat))
        modules_body.append(DynamicSpatialAttention(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# Residual Group (RG)
class DRG(nn.Module):
    def __init__(self, n_feat, out_feat, kernel_size, n_resblocks, reduction=16):
        super(DRG, self).__init__()
        modules_body = []
        modules_body = [
            RCSAB(
                default_conv, n_feat, kernel_size, reduction, bias=True, bn=True, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(default_conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

        self.conv_final = Conv(n_feat, out_feat) if n_feat != out_feat else nn.Identity()

    def forward(self, x):
        res = self.body(x)
        res += x
        return self.conv_final(res)

class C3k_DRG(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DRG(c_, c_, 3, 1) for _ in range(n)))

class C3k2_DRG(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_DRG(self.c, self.c, 2, shortcut, g) if c3k else DRG(self.c, self.c, 3, 1) for _ in range(n))
