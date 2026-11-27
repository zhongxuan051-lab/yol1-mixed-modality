# C2PSA-DyT 单层幅值调节实验

分支 `codex/experiment-dyt`，独立起点为原始混合训练基线 `4c9418e`。
采用本机包 `ultralytics/cfg/models/11/yolo11-C2PSA-DYT.yaml` 的完整配置，
仅将类别数改为 `nc=1`，文件命名 `yolo11n-dyt.yaml` 固定使用 nano 缩放。

## 结构与实验解释

仅将第 10 层 C2PSA 替换为 C2PSA_DYT。nano 下输入/输出均为 256 通道，
内部处理分支 128 通道、一个 PSABlock_DYT、Attention 为 2 heads。
在原 Attention 和 FFN 前分别增加一个 DynamicTanh：

```text
x = x + Attention(DyT1(x))
x = x + FFN(DyT2(x))
DyT(x) = weight * tanh(alpha * x) + bias
```

每个 DyT 的 alpha 是一个可训练标量，初始 0.5；weight/bias 为 128 通道
可训练参数，初始分别为 1/0。使用 NCHW 格式（channels_last=False）。
三类模块的计算和默认值均保持本机包 `extra_modules/transformer.py` 原样。
复用原 C2PSA/PSABlock 的投影、注意力、FFN 和残差，没有额外修改模块内部算法。

DyT 原工作是 CVPR 2025《Transformers without Normalization》：
<https://github.com/jiachenzhu/DyT>，<https://arxiv.org/abs/2503.10622>。
原论文研究用 DyT 替换 Transformer 归一化，本机包的 YOLO 适配是增加两处 DyT，
**保留全部现有 BN**。本实验解释为新增可学习幅值调节，不能据此声称验证了
移除归一化或 RGB/IR 专属 BN；也不代表复现原论文完整模型和结果。

其余 Backbone、Neck、P3/P4/P5 Detect、损失、TAL、NMS 沿用原始基线。
RGB 与真实 IR 混合训练无需配对，每次仍输入单张三通道图像。
不叠加 ADown、LaSEA、MALA 或 CGLU。仅使用现有 PyTorch，无额外依赖、CUDA 编译或 MMCV。

## 训练

在服务器独立实验目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用混合 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、
seed=0、amp=False、workers=6、GPU1。从 YAML 随机初始化，与基线入口一致。
数据 YAML 继续指向 `/home/ot/Students/zx/yolo11-mixed-modality`，无需复制数据。
W&B 在线初始化失败时可用 `WANDB_MODE=offline python train.py`。

输出：`falling-detection2/yolo11-real-rgb-ir-mixed-dyt-640/weights/best.pt`。
同名目录已存在时 Ultralytics 可能递增后缀；验证时用 `--weights` 指定实际路径。

## 验证与出表

```bash
python val.py
```

分别用原始基线和本实验 best.pt 在 `data_rgb.yaml`、`data.yaml` 上评估。
统一 split=val、640、batch=4、workers=0、GPU1，默认 FP32。
报告 RGB/真实 IR 的 mAP50、AP75、mAP50–95、Recall，四位小数；变化按
`(DyT−基线)×100` 个百分点计算，保留两位小数。

打印并保存 `falling-detection2/dyt-validation/comparison.md` 和 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 本地检查

```bash
python tests/check_dyt.py --smoke-train
```

追加 `--reference-pack /path/to/mianju` 可核对原包的三类 AST、完整 YAML、
C2PSA-DyT 训练/验证输出及输入和全部参数梯度。其他检查覆盖单层布局、两处
DyT 初始化、原 Attention/FFN/投影参数及 BN 保留、检测头与步长、CUDA 检测
损失/反向/AdamW、两个 DyT 的 alpha/weight/bias 梯度、640/矩形 batch=1/4
推理、Conv/BN 融合、FP16 保存重载与 FP16 推理、四次验证参数与八行表格格式
（模拟指标），以及临时合成数据一轮训练、EMA、best.pt 保存重载和独立验证。
自检关闭 W&B，使用本机 GPU0；正式入口仍为服务器 GPU1。

模型参数量 2,590,549；原始单类基线 2,590,035，增加 514 个参数。
2026-11-27 上述本地检查全部通过（含原包对照及合成数据一轮训练），环境为
Python 3.10.20、Ultralytics 8.3.6、PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop。
程序检查不代表检测精度改善；服务器真实 200 轮训练和真实 RGB/IR 验证尚未执行。

## 离线 Git 备用

配套 `dyt.bundle` 是基于 `4c9418e` 的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/dyt.bundle
git fetch /path/to/dyt.bundle codex/experiment-dyt:refs/remotes/offline/experiment-dyt
git worktree add -b codex/experiment-dyt \
  /home/ot/Students/zx/yolo11-mixed-modality-dyt \
  refs/remotes/offline/experiment-dyt
```

进入新目录后执行训练命令即可。
