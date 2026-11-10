# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Grouped dynamic upsampling with the LP offsets from https://arxiv.org/abs/2308.15085."""

    def __init__(self, channels, scale=2, groups=4):
        super().__init__()
        if scale < 1 or groups < 1 or channels < groups or channels % groups:
            raise ValueError("DySample requires a positive scale and channels divisible by groups")
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(channels, 2 * groups * scale**2, kernel_size=1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        out_h, out_w = height * self.scale, width * self.scale
        offsets = F.pixel_shuffle(self.offset(x), self.scale).to(x.dtype)
        offset_x, offset_y = offsets.reshape(batch, 2, self.groups, out_h, out_w).unbind(dim=1)

        # Start from bilinear pixel centers and apply LP offsets scaled by 0.25 input pixels.
        center_x = (2 * torch.arange(out_w, device=x.device, dtype=x.dtype) + 1) / out_w - 1
        center_y = (2 * torch.arange(out_h, device=x.device, dtype=x.dtype) + 1) / out_h - 1
        sample_x = center_x + offset_x * (0.5 / width)
        sample_y = center_y[:, None] + offset_y * (0.5 / height)
        grid = torch.stack((sample_x, sample_y), dim=-1).flatten(0, 1)
        features = x.reshape(batch * self.groups, channels // self.groups, height, width)
        sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return sampled.reshape(batch, channels, out_h, out_w)
