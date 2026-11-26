# Ultralytics integration: AGPL-3.0 license
# DCNv4 / CenterFeatureScaleModule: Copyright (c) 2023 OpenGVLab, MIT license.
"""DCNv4 from the supplied mianju pack, with unchanged module computations.

DCNv4 (CVPR 2024): https://github.com/OpenGVLab/DCNv4
The DCNv4/CenterFeatureScale classes retain OpenGVLab's MIT-licensed design.
CPU uses a bilinear reference for Ultralytics' stride initialization.
CUDA always requires the supplied compiled extension; no silent substitution.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import math

from .block import Bottleneck, C3k, C3k2
from .conv import Conv, autopad
from .dcnv4_backend import DCNv4Function

__all__ = ("DCNv4", "DCNV4_YOLO", "Bottleneck_DCNV4", "C3k_DCNv4", "C3k2_DCNv4")

class CenterFeatureScaleModule(nn.Module):

    def forward(self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query, weight=center_feature_scale_proj_weight, bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv4(nn.Module):

    def __init__(self, channels=64, kernel_size=3, stride=1, pad=1, dilation=1, group=4, offset_scale=1.0, dw_kernel_size=None, center_feature_scale=False, remove_center=False, output_bias=True, without_pointwise=False, **kwargs):
        """
        DCNv4 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise
        self.K = group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil(self.K * 3 / 8) * 8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.0)
        constant_(self.offset_mask.bias.data, 0.0)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.0)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input, shape=None):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        input = input.reshape((input.size(0), input.size(1), input.size(2) * input.size(3))).transpose(2, 1)
        (N, L, C) = input.shape
        if shape is not None:
            (H, W) = shape
        else:
            (H, W) = (int(L ** 0.5), int(L ** 0.5))
        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input.view(N, H, W, C).permute(0, 3, 1, 2))
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)
        x_proj = x
        x = DCNv4Function.apply(x, offset_mask, self.kernel_size, self.kernel_size, self.stride, self.stride, self.pad, self.pad, self.dilation, self.dilation, self.group, self.group_channels, self.offset_scale, 256, self.remove_center)
        x = x.view(N, L // self.stride // self.stride, -1)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        if not self.without_pointwise:
            x = self.output_proj(x)
        return x.transpose(2, 1).reshape((N, C, H // self.stride, W // self.stride))


class DCNV4_YOLO(nn.Module):

    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = self.dcnv4(x, (x.size(2), x.size(3)))
        x = self.act(self.bn(x))
        return x


class Bottleneck_DCNV4(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv2 = DCNV4_YOLO(c_, c2, k[1])


class C3k_DCNv4(C3k):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_DCNV4(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_DCNv4(C3k2):

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList((C3k_DCNv4(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_DCNV4(self.c, self.c, shortcut, g) for _ in range(n)))
