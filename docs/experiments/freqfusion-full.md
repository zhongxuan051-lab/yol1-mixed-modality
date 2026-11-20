# Full FreqFFPN Neck for RGB/IR mixed detection

Baseline: `4c9418e`, standard YOLO11n mixed training. This branch evaluates the
complete supplied `yolo11-FreqFFPN.yaml`, not a cumulative change on FDConv.

## Rationale and scope

Frequency-aware fusion addresses feature inconsistency and boundary displacement
when combining high- and low-resolution feature maps. Whether this improves
the small-person detections in the RGB/real-IR dataset remains to be tested.

- Paper: Frequency-aware Feature Fusion for Dense Image Prediction, TPAMI 2024.
  https://arxiv.org/abs/2408.12879
- Author implementation: https://github.com/Linwei-Chen/FreqFusion
- Port source: supplied module pack `FreqFusion.py` and `yolo11-FreqFFPN.yaml`.

The backbone stays standard. Three 1x1 Conv projections align the P3/P4/P5
features to 64 channels in the n scale. Two FreqFusion modules perform the
top-down P5-to-P4 and P4-to-P3 fusion by summing filtered features. The remaining
bottom-up path and C3k2 widths follow the complete pack configuration. Therefore
this experiment includes Neck topology and channel-width changes, not just the
insertion of one operator. The detector still takes one three-channel image and
predicts on P3/P4/P5; RGB/IR pairing is not required.

## Exact module settings

The package defaults are preserved: adaptive low-pass kernel 5, adaptive high-pass
kernel 3, semi-convolution initialization, Hamming regularization, and a residual
high-resolution path. The compressed width is `(hr_channels + lr_channels) // 8`,
16 in this experiment. **Offset resampling is disabled**, matching the supplied
configuration. This run must not be described as testing that optional component.
The dormant sampler's input-width argument is corrected to use the actual
compressed width; this does not affect the default experiment.

The pack depends on MMCV's CARAFE and Xavier initializer without a working
fallback. The port supplies a pure PyTorch zero-padded CARAFE reassembly backend
and the initializer, avoiding compiled third-party operators. Native-resolution
unfolded patches are contracted with output masks before pixel rearrangement,
avoiding explicit duplication of all patches at upsampled resolution. All kernel
normalization, filter generators and module settings remain from the pack.
Unused comments are removed; no preceding loss or sampling experiments are used.

## Training and evaluation

Same baseline conditions: random initialization, `data_mixed.yaml`, image size
640, batch 32, 200 epochs, AdamW, lr0=0.001, seed 0, AMP off, workers 6, GPU 1.
Run: `falling-detection2/yolo11-real-rgb-ir-mixed-freqfusion-full-640`.

```bash
cd /home/ot/Students/zx/yolo11-mixed-modality
git fetch origin
git worktree add -b codex/experiment-freqfusion-full /home/ot/Students/zx/yolo11-mixed-modality-freqfusion-full origin/codex/experiment-freqfusion-full
cd /home/ot/Students/zx/yolo11-mixed-modality-freqfusion-full
python train.py
```

The dataset YAML retains the existing absolute path under the main checkout.
No dataset duplication or new dependencies are required.

After training, `python val.py` compares baseline and FreqFusion best.pt on both
validation sets: 640, batch 4, workers 0, GPU 1. It prints the full precision,
recall, AP50, AP75 and mAP50-95 results, followed by a per-metric comparison table
with the baseline values and changes in percentage points. Test sets are reserved.

## Local integration checks

- Two FreqFusion modules; 2,476,087 parameters before fusion, 2,468,383 after fusion.
- Independent CARAFE reference checked forward results and gradients with
  respect to features and masks for scales 1/2 and channel groups 1/2.
- 640 CUDA inference, real YOLO loss/backward/AdamW, and nonzero finite gradients
  in both modules' low/high-pass generators passed.
- Rectangular batch-one inference, BN-fusion consistency, half checkpoint reload
  and half inference passed.
- One synthetic epoch through the actual trainer, EMA, checkpoint saving and
  validation passed, including fresh FP16 batch-one checkpoint validation.

These checks establish integration, not target-dataset accuracy or runtime speed.
Built-in THOP estimates can omit functional reassembly work; measure latency and
memory on the target device before making deployment-efficiency claims.
