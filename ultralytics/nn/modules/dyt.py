# Ultralytics YOLO, AGPL-3.0 license
"""Pack-default DyT additions before C2PSA Attention and FFN.

Dynamic Tanh: Transformers without Normalization (CVPR 2025).
https://github.com/jiachenzhu/DyT
This YOLO adaptation adds DyT operations and retains all original BN layers.
The three class computations/defaults are unchanged from the mianju pack.
"""
import torch
import torch.nn as nn

from .block import C2PSA, PSABlock

__all__ = ("DynamicTanh", "PSABlock_DYT", "C2PSA_DYT")

class DynamicTanh(nn.Module):

    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}'


class PSABlock_DYT(PSABlock):

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)

    def forward(self, x):
        x = x + self.attn(self.dyt1(x)) if self.add else self.attn(self.dyt1(x))
        x = x + self.ffn(self.dyt2(x)) if self.add else self.ffn(self.dyt2(x))
        return x


class C2PSA_DYT(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_DYT(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
