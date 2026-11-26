"""CPU reference for model initialization; strict pack CUDA backend on GPU."""
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=1)
def require_cuda_extension():
    """Fail before training if the compiled extension cannot be loaded."""
    if not torch.cuda.is_available():
        raise RuntimeError("DCNv4 training requires CUDA. Run python scripts/install_dcnv4.py on the GPU server.")
    try:
        from DCNv4.functions.dcnv4_func import DCNv4Function as compiled
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Cannot load the supplied DCNv4 CUDA extension. Build it in this Python environment with "
            "python scripts/install_dcnv4.py, then run python tests/check_dcnv4.py --smoke-train."
        ) from error
    return compiled


def dcnv4_reference(value, offset_mask, kernel_h, kernel_w, stride_h, stride_w,
                    pad_h, pad_w, dilation_h, dilation_w, group, group_channels,
                    offset_scale, im2col_step, remove_center):
    """Bilinear reference matching the pack kernel's NHWC and per-group packing.

    Per group: K interleaved (x,y) offsets followed by K raw weights. Kernel
    points enumerate x first, then y. Rounded-up offset channels are padding.
    This path allows Ultralytics' CPU dummy forward to calculate strides.
    """
    batch, height, width, channels = value.shape
    out_h = (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    if channels != group * group_channels or offset_mask.shape[:3] != (batch, out_h, out_w):
        raise ValueError("Invalid DCNv4 input/offset shape")
    points = [(x, y) for x in range(kernel_w) for y in range(kernel_h)
              if not (remove_center and x == kernel_w // 2 and y == kernel_h // 2)]
    count = len(points)
    dtype = value.dtype
    if dtype in (torch.float16, torch.bfloat16):
        value, offset_mask = value.float(), offset_mask.float()
    packed = offset_mask[..., :group * count * 3].reshape(batch, out_h, out_w, group, count * 3)
    offsets = packed[..., :count * 2].reshape(batch, out_h, out_w, group, count, 2)
    weights = packed[..., count * 2:]
    yy, xx = torch.meshgrid(torch.arange(out_h, device=value.device, dtype=value.dtype),
                            torch.arange(out_w, device=value.device, dtype=value.dtype), indexing="ij")
    cx, cy = dilation_w * (kernel_w - 1) // 2, dilation_h * (kernel_h - 1) // 2
    centers = torch.stack((xx * stride_w + cx - pad_w, yy * stride_h + cy - pad_h), -1)
    stencil = value.new_tensor([(x * dilation_w - cx, y * dilation_h - cy) for x, y in points])
    locations = centers[None, :, :, None, None, :] + offset_scale * (stencil + offsets)
    grid = (locations + .5) * value.new_tensor([2 / width, 2 / height]) - 1
    grid = grid.permute(0, 3, 1, 2, 4, 5).reshape(batch * group, out_h * out_w, count, 2)
    image = value.permute(0, 3, 1, 2).reshape(batch * group, group_channels, height, width)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    weights = weights.permute(0, 3, 1, 2, 4).reshape(batch * group, 1, out_h * out_w, count)
    result = (sampled * weights).sum(-1).reshape(batch, channels, out_h, out_w)
    return result.permute(0, 2, 3, 1).contiguous().to(dtype)


class DCNv4Function:
    """Preserve the pack's .apply interface without hiding missing CUDA builds."""

    @staticmethod
    def apply(value, offset_mask, *args):
        if value.is_cuda:
            return require_cuda_extension().apply(value, offset_mask, *args)
        return dcnv4_reference(value, offset_mask, *args)
