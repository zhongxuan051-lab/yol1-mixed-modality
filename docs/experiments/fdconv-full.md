# YOLO11n RGB/IR mixed training with full C3k2-FDConv

This experiment starts from the fixed mixed-training baseline `4c9418e`.
It follows the supplied `yolo11-C3k2-FDConv.yaml`: all eight C3k2 blocks at
layers 2, 4, 6, 8, 13, 16, 19 and 22 use C3k2_FDConv. Detection remains
at P3/P4/P5 with one three-channel image per input.

## Motivation and scope

FDConv learns frequency-diverse kernel weights and adapts their responses
using kernel spatial modulation and frequency-band modulation. This experiment
tests whether that adaptability helps a shared RGB/real-IR detector. Improved
performance on this dataset is a hypothesis, not a reported result of FDConv.

- Paper: Frequency Dynamic Convolution for Dense Image Prediction, CVPR 2025.
  https://arxiv.org/abs/2503.18783
- Author code: https://github.com/Linwei-Chen/FDConv
- Port source: supplied module pack, `FDConv_initialversion.py`, plus its three
  C3k2 wrapper classes. The pack's raw FDConv bottleneck replacements are
  retained without adding extra BN or activation layers.
- Default settings are retained: 16 kernel groups, parameter ratio 1,
  frequency bands [2, 4, 8], global/local kernel modulation and band modulation.
- There are 22 FDConv objects in the n model. The pack falls back to ordinary
  convolution when either channel count is <=16, so 16 use dynamic kernels
  and six use that intended fallback.

Compatibility changes: remove unused imports and an unfinished profiling stub;
register frequency indices as nonpersistent buffers for device transfer; use
explicit meshgrid indexing. The global pooled BN uses running statistics when
only one value per channel is available, including YOLO's batch-one stride
initialization. Ordinary training batches retain normal BN behavior.

Functional FFT and dynamic convolution operations may be missed by the built-in
THOP profiler. Do not use its printed GFLOPs alone to compare runtime cost.

## Training

Same baseline training: random initialization, mixed training data, image size
640, batch 32, 200 epochs, AdamW, lr0=0.001, seed 0, AMP disabled, workers 6.
GPU is 1. The run is `yolo11-real-rgb-ir-mixed-fdconv-full-640`.
No preceding experimental sampling, assigner or regression changes are included.

From the existing main server checkout:

```bash
cd /home/ot/Students/zx/yolo11-mixed-modality
git fetch origin
git worktree add -b codex/experiment-fdconv-full /home/ot/Students/zx/yolo11-mixed-modality-fdconv-full origin/codex/experiment-fdconv-full
cd /home/ot/Students/zx/yolo11-mixed-modality-fdconv-full
python train.py
```

The data YAML keeps the existing absolute dataset root under the main checkout;
dataset files do not need to be duplicated.

## Validation

After training, run `python val.py` from the FDConv worktree. It evaluates both
best checkpoints on the same RGB/IR validation sets with image size 640, batch 4,
workers 0 and GPU 1, printing P, R, AP50, AP75 and mAP50-95.
Keep the test split reserved for final evaluation after model selection.

## Local integration verification

- Eight C3k2-FDConv blocks; 2,700,027 parameters before BN fusion.
- CUDA float32 inference at 640 and rectangular batch-one inference passed.
- YOLO detection loss, finite gradients in all 16 dynamic kernel coefficient
  tensors, and an AdamW step passed.
- BN fusion output consistency and half-precision checkpoint round trip passed.
- A one-epoch synthetic run through the actual trainer, EMA, checkpoint saving,
  end-of-training validation, and fresh FP16 batch-one validation passed.

These checks establish integration; they do not measure detection accuracy.
