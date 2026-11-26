# ADown 五处下采样实验

分支 `codex/experiment-adown`，独立起点为原始混合训练基线 `4c9418e`。
采用本机包 `ultralytics/cfg/models/11/yolo11-ADown.yaml` 的完整配置，
仅将类别数改为 `nc=1`，文件命名 `yolo11n-adown.yaml` 固定使用 nano 缩放。

## 改动范围

将以下 5 处步长为 2 的卷积替换为 ADown，编号从 0 开始：

| 位置 | 输入→输出通道（nano） | 输出尺度 |
|---|---|---|
| Backbone 3 | 64→64 | P3/8 |
| Backbone 5 | 128→128 | P4/16 |
| Backbone 7 | 128→256 | P5/32 |
| Neck 17 | 64→64 | P4/16 |
| Neck 20 | 128→128 | P5/32 |

保留前两层普通卷积、全部 C3k2、SPPF、C2PSA、Concat、上采样、Detect，
不叠加 LaSEA 或其他实验。损失、TAL、NMS 和输入形式沿用基线：RGB 与真实 IR
混合训练，无需配对，推理时每次输入单张三通道图像。

ADown 先执行 2×2、stride=1 的平均池化，再沿通道均分：一路用 3×3、
stride=2 卷积，另一路用 3×3、stride=2 最大池化后接 1×1 卷积，最终通道拼接。
沿用包内池化参数、Conv/BN/SiLU 默认设置。

当前 Ultralytics 8.3.6 已包含同一 ADown 实现及模型解析注册；直接复用
`ultralytics/nn/modules/block.py`，无需修改底层库、额外依赖或 CUDA 编译。
已对照本机包 `extra_modules/block.py` 核对实际输出和所有梯度，计算完全一致。
来源为 YOLOv9（ECCV 2024）：<https://github.com/WongKinYiu/yolov9>。
本实验只移植 ADown 配置，不代表采用完整 YOLOv9、GELAN 或 PGI 训练方法。

## 训练

在服务器独立实验目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、seed=0、
amp=False、workers=6、GPU1，从 YAML 随机初始化，与原始基线入口一致。
数据 YAML 继续使用服务器主目录内的数据，无需复制数据。
W&B 在线初始化失败时可用 `WANDB_MODE=offline python train.py`。

输出：`falling-detection2/yolo11-real-rgb-ir-mixed-adown-640/weights/best.pt`。
如果同名目录已经存在，Ultralytics 可能递增后缀；验证时用 `--weights` 指向实际文件。

## 验证与出表

```bash
python val.py
```

分别评估原始混合基线和 ADown 的 best.pt。RGB 用 `data_rgb.yaml`，真实 IR 用
`data.yaml`，统一 split=val、640、batch=4、workers=0、GPU1，默认 FP32。
报告 mAP50、AP75、mAP50–95、Recall，指标四位小数，变化为
`(ADown−基线)×100` 个百分点，两位小数。

输出打印并保存到 `falling-detection2/adown-validation/comparison.md` 和
`comparison.csv`。支持 `--dataset-root`、`--weights`、`--baseline-weights`、
`--device`、`--output-dir`。

## 本地检查

```bash
python tests/check_adown.py --smoke-train
```

追加 `--reference-pack /path/to/mianju` 可核对本机包原始模块和完整 YAML。
检查覆盖五个通道组合下原包与内置 ADown 的训练/验证输出及所有梯度、五处结构
替换、原检测头与步长、CUDA 检测损失/反向/AdamW 和两分支梯度、640/矩形
batch=1/4 推理、Conv/BN 融合、FP16 权重保存重载和 FP16 推理、四次真实验证
调用参数与八行出表格式（模拟指标）。可选合成数据一轮训练覆盖 Trainer、EMA、
best.pt 保存重载及独立验证。自检关闭 W&B，使用本机 GPU0 和临时合成数据。

模型参数量 2,110,803；原始单类基线 2,590,035，减少 479,232。
参数减少不代表检测精度提高，仍以 RGB/IR 验证结果决定是否保留。

2026-11-26 上述检查含原包对照及合成数据一轮训练全部通过。环境为 Python
3.10.20、Ultralytics 8.3.6、PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop。
未执行服务器真实 200 轮训练或真实数据精度评估。

## 离线 Git 备用

配套 `adown.bundle` 为基于 `4c9418e` 的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/adown.bundle
git fetch /path/to/adown.bundle codex/experiment-adown:refs/remotes/offline/experiment-adown
git worktree add -b codex/experiment-adown \
  /home/ot/Students/zx/yolo11-mixed-modality-adown \
  refs/remotes/offline/experiment-adown
```

进入新目录后执行训练命令即可，无需编译扩展。
