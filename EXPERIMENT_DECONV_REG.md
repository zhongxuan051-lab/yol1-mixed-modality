# DEConv 三尺度框回归实验

分支 `codex/experiment-deconv-reg`，起点为原始混合训练基线 `4c9418e`。
将第 23 层 Detect 的三个框回归支路中第二个 3×3 卷积分别替换为 DEConv。
具体位置为 P3 的 `cv2[0][1]`、P4 的 `cv2[1][1]`、P5 的 `cv2[2][1]`，
三处都是 64→64 通道。继续使用原 Detect 的分类支路、DFL、前向与解码逻辑，
保留训练损失、标签分配及 NMS。

DEConv 在前向时将普通卷积、中心差分、水平差分、垂直差分和角度差分
的权重及偏置合并为一个 3×3 卷积，再执行 BN 和 SiLU。本实验移植本机包
`extra_modules/deconv.py` 的 DEConv 算子，训练与推理时通过原生 PyTorch
完成核重排，不需要 einops 或额外 CUDA 编译。算子来源为
[DEA-Net（IEEE TIP 2024）](https://github.com/cecret3350/DEA-Net)。

## 训练

```bash
WANDB_MODE=online python train.py
```

混合数据 `data_mixed.yaml`，单张三通道输入，640、batch=32、200 轮、
AdamW、lr0=0.001、seed=0、AMP 关闭、workers=6、GPU1，从 YAML 初始化。
权重位于 `falling-detection2/yolo11-real-rgb-ir-mixed-deconv-reg-640/weights/best.pt`。
如训练目录增加数字后缀，验证时用 `--weights` 指定实际权重文件。

## 验证

```bash
python val.py
```

基线和 DEConv 各自的 best.pt 分别在 `data_rgb.yaml`（RGB）和 `data.yaml`
（真实 IR）上验证，统一 split=val、640、batch=4、workers=0、GPU1。
在 `falling-detection2/deconv-reg-validation/` 中输出：

- `comparison.md`、`comparison.csv`：mAP50、AP75、mAP50–95、Recall 及百分点变化。
- `size_recall.md`：沿用 DRFD 实验的每图置信度降序一对一 IoU≥0.50 匹配，
  按 640 输入上的标注框短边分组。两个模型在每种模态默认使用相同的 216 个误检上限，
  并记录实际误检数及阈值。可用 `--fp-budget-rgb`、`--fp-budget-ir` 分别调整。

重点比较真实 IR 的 AP75、mAP50–95，以及两种模态下 8–<16 px 组的召回。

## Git 离线备用

将 `deconv-reg.bundle` 传到服务器后，在主仓库执行：

```bash
git bundle verify /path/to/deconv-reg.bundle
git fetch /path/to/deconv-reg.bundle codex/experiment-deconv-reg:refs/remotes/offline/experiment-deconv-reg
git worktree add -b codex/experiment-deconv-reg \
  /home/ot/Students/zx/yolo11-mixed-modality-deconv-reg \
  refs/remotes/offline/experiment-deconv-reg
```

进入新目录执行训练命令。
