# Ultralytics YOLO, AGPL-3.0 license
"""C2PSA Convolutional GLU from the supplied mianju module pack.

TransNeXt (CVPR 2024): https://arxiv.org/abs/2311.17132
The GLU returns only its transformed features. PSABlock supplies the single
outer FFN residual, avoiding the pack's combined 2*x + F(x) behavior.
Other defaults and the single-stage YAML are retained.
"""
import torch.nn as nn

from .block import C2PSA, PSABlock

__all__ = ("ConvolutionalGLU", "PSABlock_CGLU", "C2PSA_CGLU")

class ConvolutionalGLU(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features), act_layer())
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        (x, v) = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PSABlock_CGLU(PSABlock):

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.ffn = ConvolutionalGLU(c, c * 2, c)


class C2PSA_CGLU(C2PSA):

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(*(PSABlock_CGLU(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
