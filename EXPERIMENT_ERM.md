# ERM 四处融合实验

分支：`codex/experiment-erm-full`。共同起点：原始混合训练基线 `4c9418e`。
采用本机包 `ultralytics/cfg/models/11/yolo11-ERM.yaml` 的完整四处布局，
类别数沿用基线 `nc=1`。这是独立实验，LaSEA 继续保留为另一候选。

## 改动范围

将 Neck 第 **12、15、18、21** 层的 Concat 换为 ERM，层号从 0 开始。

| 层号 | 融合位置 | nano 实际输入通道 | 融合输出通道 |
|---|---|---|---|
| 12 | 上采样深层特征与 Backbone P4 | 256、128 | 128 |
| 15 | 上采样特征与 Backbone P3 | 128、128 | 64 |
| 18 | 下采样 P3 与 Neck P4 | 64、128 | 128 |
| 21 | 下采样 P4 与 Backbone P5 | 128、256 | 128 |

融合输出通道及后续 C3k2 的输入通道按包内配置适配；Backbone、C3k2 的配置参数、
下采样、SPPF、C2PSA、Detect、损失、TAL 和 NMS 沿用基线。
RGB 与真实 IR 混合训练无需配对，每次仍输入一张三通道图像。

模块来源：本机包 `extra_modules/block.py` 中的 ERM、ExternalAttention、
get_sobel、run_sobel。原工作为 ISGLNet：<https://github.com/fuqingzhang/ISGLNet>。
这是 ERM 移植实验，不是整套 ISGLNet 的复现。

## 机制与明确修正

两路特征投影后相加，通过卷积与 ExternalAttention，再提取 Sobel 边缘响应。
根据学习到的分数构建前景与模糊区域权重，聚合两个区域的上下文，经 query/key/value
交互生成细化权重。最终输出对第二路投影特征的加权结果。无需额外边缘标注或损失。

接入时保留包内投影、注意力、固定 Sobel 卷积权重及其 BN，并作三项明确修正：

1. 四个 `Conv(..., act=nn.ReLU)` 调用改为 `act=nn.ReLU()`，使激活实际生效。
2. 区域聚合前，原 `x.view(B, H*W, -1)` 改为 `x.flatten(2).transpose(1,2)`，
   正确建立每个像素与各通道的对应关系。
3. Sobel 幅值改为 `sqrt(gx²+gy²+1e-6)`，避免零梯度位置出现未定义导数。
   本地检查复现了原 `sqrt(0)` 路径的非有限梯度，并确认修正后有限。

与包内参考的数值/梯度对照同步应用上述修正；本实验不宣称与未经修正的原包输出一致。
四处布局和其余算子/默认值保持包内设置。

## 训练与验证

在服务器独立工作目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、seed=0、
amp=False、workers=6、GPU1。从 YAML 随机初始化，与原基线训练入口一致。
数据 YAML 仍指向 `/home/ot/Students/zx/yolo11-mixed-modality`，无需复制数据。
只使用现有 PyTorch 和 NumPy，不需要额外依赖或编译扩展。
W&B 在线初始化失败时，可用 `WANDB_MODE=offline python train.py`。

训练输出：`falling-detection2/yolo11-real-rgb-ir-mixed-erm-full-640/weights/best.pt`。
同名运行目录存在时 Ultralytics 可能增加后缀；验证用 `--weights` 指定实际权重。

训练后执行 `python val.py`，分别评估原始混合基线与 ERM 的 best.pt：
RGB 用 `data_rgb.yaml`，真实 IR 用 `data.yaml`，统一 `split=val`、640、
batch=4、workers=0、GPU1。每个模态报告 mAP50、AP75、mAP50–95、Recall，
四位小数。变化为 `(ERM−基线)×100` 个百分点。
表格保存到 `falling-detection2/erm-validation/comparison.md` 与 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 本地检查

运行 `python tests/check_erm.py --smoke-train`；可追加 `--reference-pack PATH`
直接核对用户包（同步修正上述三个问题）。
检查覆盖独立区域加权聚合、零边缘反向稳定性、四处布局与通道、训练/验证输出及
全部梯度、CUDA 检测损失及 AdamW、640/矩形和 batch=1/4 推理、Conv/BN 融合、
FP16 保存重载、四项指标出表，以及合成数据一轮训练、EMA、best.pt 保存及重载验证。

2026-11-25 上述检查全部通过，环境为 Ultralytics 8.3.6、Python 3.10.20、
PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop GPU。参数量为 4,498,599。
合成数据仅验证程序流程；服务器 200 轮训练和真实 RGB/IR 验证尚未执行。

## 离线 Git 备用

配套 `erm-full.bundle` 是基于 `4c9418e` 的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/erm-full.bundle
git fetch /path/to/erm-full.bundle codex/experiment-erm-full:refs/remotes/offline/experiment-erm-full
git worktree add -b codex/experiment-erm-full \
  /home/ot/Students/zx/yolo11-mixed-modality-erm-full \
  refs/remotes/offline/experiment-erm-full
```
