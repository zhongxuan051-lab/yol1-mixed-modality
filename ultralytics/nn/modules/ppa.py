# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Parallelized patch-aware attention for YOLO11, based on HCF-Net (arXiv:2403.10778)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .conv import Conv


class PatchAttention(nn.Module):
    """Select local patch features with a learned task embedding."""

    def __init__(self, channels, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Linear(patch_size**2, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.Linear(channels // 2, channels),
        )
        self.prompt = nn.Parameter(torch.randn(channels))
        self.channel_mix = nn.Parameter(torch.eye(channels))
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, channel_mean):
        height, width = channel_mean.shape[-2:]
        size = self.patch_size
        # Pad partial patches so rectangular feature maps retain their border pixels.
        padded = F.pad(channel_mean, (0, -width % size, 0, -height % size))
        patches = F.pixel_unshuffle(padded, size).permute(0, 2, 3, 1)
        tokens = self.encoder(patches)
        tokens = tokens * tokens.softmax(dim=-1)
        relevance = F.cosine_similarity(tokens, self.prompt, dim=-1, eps=1e-6).clamp(0, 1)
        selected = (tokens * relevance.unsqueeze(-1)) @ self.channel_mix
        attention = F.interpolate(
            selected.permute(0, 3, 1, 2), size=padded.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.projection(attention[..., :height, :width])


class PPA(nn.Module):
    """Fuse two patch scales and a serial convolution branch, then apply channel and spatial attention."""

    def __init__(self, channels):
        super().__init__()
        self.skip = Conv(channels, channels, act=False)
        self.small_patches = PatchAttention(channels, patch_size=2)
        self.large_patches = PatchAttention(channels, patch_size=4)
        self.conv1 = Conv(channels, channels, 3)
        self.conv2 = Conv(channels, channels, 3)
        self.conv3 = Conv(channels, channels, 3)
        kernel = int((math.log2(channels) + 1) / 2)
        kernel = 2 * (kernel // 2) + 1
        self.channel_gate = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)
        self.spatial_gate = nn.Conv2d(2, 1, 7, padding=3)
        self.output = nn.Sequential(nn.Dropout2d(0.1), nn.BatchNorm2d(channels), nn.SiLU())

    def forward(self, x):
        skip = self.skip(x)
        channel_mean = skip.mean(dim=1, keepdim=True)
        first = self.conv1(x)
        second = self.conv2(first)
        third = self.conv3(second)
        features = first + second + third + skip + self.small_patches(channel_mean) + self.large_patches(channel_mean)

        channel_weights = self.channel_gate(features.mean(dim=(2, 3)).unsqueeze(1)).sigmoid()
        features = features * channel_weights.squeeze(1)[..., None, None]
        spatial_stats = torch.cat((features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)), dim=1)
        return self.output(features * self.spatial_gate(spatial_stats).sigmoid())


class C3k2_PPA(C3k2):
    """Use PPA in place of the inner bottlenecks at both C3k2 depths."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        if c3k:
            for block in self.m:
                channels = block.cv1.conv.out_channels
                block.m = nn.Sequential(*(PPA(channels) for _ in block.m))
        else:
            self.m = nn.ModuleList(PPA(self.c) for _ in range(n))
