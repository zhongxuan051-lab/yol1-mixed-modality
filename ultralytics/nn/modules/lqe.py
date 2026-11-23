# Ultralytics YOLO, AGPL-3.0 license
"""Distribution-statistics quality correction for the baseline YOLO11 Detect head.

Adapted from the supplied module pack's Detect_LQE (GFLV2-inspired LQE):
https://arxiv.org/abs/2011.12885
Only the head is changed; classification logits receive an additive correction.
This is not a reproduction of the full GFLV2 detector or its losses.
"""

import torch
from torch import nn

from .head import Detect

__all__ = ("LQE", "Detect_LQE")


class LQE(nn.Module):
    """Predict a scalar logit correction from each edge's top-k bin probabilities."""

    def __init__(self, reg_max=16, k=4, hidden_dim=64):
        super().__init__()
        if not 1 <= k <= reg_max:
            raise ValueError("LQE requires 1 <= k <= reg_max")
        self.reg_max = reg_max
        self.k = k
        self.reg_conf = nn.Sequential(
            nn.Conv2d(4 * (k + 1), hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Start with the unmodified baseline logits, as in the supplied pack.
        nn.init.zeros_(self.reg_conf[-1].weight)
        nn.init.zeros_(self.reg_conf[-1].bias)

    def distribution_statistics(self, pred_corners):
        """Respect DFL's contiguous [left, top, right, bottom] groups of bins."""
        b, _, h, w = pred_corners.shape
        # YOLO DFL uses [B, 4, bins, H, W], not [B, bins, 4, H, W].
        probabilities = pred_corners.reshape(b, 4, self.reg_max, h, w).softmax(dim=2)
        topk = probabilities.topk(self.k, dim=2).values
        statistics = torch.cat((topk, topk.mean(dim=2, keepdim=True)), dim=2)
        return statistics.reshape(b, 4 * (self.k + 1), h, w)

    def forward(self, scores, pred_corners):
        return scores + self.reg_conf(self.distribution_statistics(pred_corners))


class Detect_LQE(Detect):
    """Baseline P3/P4/P5 Detect with an independent LQE branch at each scale."""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        self.lqe = nn.ModuleList(LQE(reg_max=self.reg_max) for _ in ch)

    def forward(self, x):
        for i in range(self.nl):
            corners = self.cv2[i](x[i])
            scores = self.lqe[i](self.cv3[i](x[i]), corners)
            x[i] = torch.cat((corners, scores), dim=1)
        if self.training:
            return x
        predictions = self._inference(x)
        return predictions if self.export else (predictions, x)
