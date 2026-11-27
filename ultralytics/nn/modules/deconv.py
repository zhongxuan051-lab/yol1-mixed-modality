"""Detail-enhanced convolution from DEA-Net, adapted from the local module pack."""

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv


class DEConv(nn.Module):
    """Combine a regular kernel and four learnable difference kernels."""

    def __init__(self, channels):
        super().__init__()
        self.center = nn.Conv2d(channels, channels, 3, bias=True)
        self.horizontal = nn.Conv1d(channels, channels, 3, bias=True)
        self.vertical = nn.Conv1d(channels, channels, 3, bias=True)
        self.angular = nn.Conv2d(channels, channels, 3, bias=True)
        self.regular = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(channels)
        self.act = Conv.default_act

    def forward(self, x):
        center = self.center.weight.flatten(2)
        center_diff = center.clone()
        center_diff[..., 4] = center[..., 4] - center.sum(dim=-1)

        horizontal = self.horizontal.weight
        horizontal_diff = torch.stack((horizontal, torch.zeros_like(horizontal), -horizontal), dim=-1)
        vertical = self.vertical.weight
        vertical_diff = torch.stack((vertical, torch.zeros_like(vertical), -vertical), dim=-2)

        angular = self.angular.weight.flatten(2)
        angular_diff = angular - angular[..., [3, 0, 1, 6, 4, 2, 7, 8, 5]]

        weight = (
            center_diff.reshape_as(self.regular.weight)
            + horizontal_diff
            + vertical_diff
            + angular_diff.reshape_as(self.regular.weight)
            + self.regular.weight
        )
        bias = self.center.bias + self.horizontal.bias + self.vertical.bias + self.angular.bias + self.regular.bias
        return self.act(self.bn(F.conv2d(x, weight, bias, padding=1)))
