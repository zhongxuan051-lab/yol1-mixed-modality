# Ultralytics YOLO, AGPL-3.0 license
"""FSConv: exact module-pack spatial/wavelet downsampling, including SEAttention.

FSCFNet, TGRS 2025: https://github.com/qzxwj/FSCFNet
Adapted from the supplied mianju module pack's block.py and attention.py.
DWTForward(J=1, mode='zero', wave='haar') is retained, not approximated.
Install the experiment dependencies with: pip install -r requirements-fsconv.txt
"""
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from .conv import Conv

__all__ = ("FSConv",)

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FSConv(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        from pytorch_wavelets import DWTForward
        self.c1 = c1

        self.conv1 = Conv(c1, 2 * c1, 1, g=c1)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv2 = Conv(c1, c2, k, 2, g=math.gcd(c1, c2))
        self.conv3 = Conv(c1 * 3, c2, 3, d=1, g=math.gcd(c1 * 3, c2))
        self.se = SEAttention(c2)
        self.conv4 = Conv(c1, c2, 3, g=math.gcd(c1, c2))
        self.conv5 = Conv(2 * c2, c2, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1, x2 = torch.split(x0, self.c1, dim=1)
        conv_spatial = self.conv2(x1)

        yL, yH = self.wt(x2)

        # Extract the high-frequency subbands
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]

        high_frequency_fused = torch.cat([y_HL, y_LH, y_HH], dim=1)
        high_frequency_fused_output = self.conv3(high_frequency_fused)

        # Apply SE attention
        high_frequency_fused_output = self.se(high_frequency_fused_output)

        low_frequency_fused_output = self.conv4(yL)

        spatial_output = conv_spatial * high_frequency_fused_output

        fused = torch.cat([spatial_output, low_frequency_fused_output], dim=1)
        out = self.conv5(fused)

        return out
