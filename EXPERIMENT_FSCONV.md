# FSConv 全七处下采样实验

分支：`codex/experiment-fsconv-full`。共同起点：原始混合训练基线 `4c9418e`。
这次按用户指定采用本机包的完整 FSConv YAML，单类 `nc=1`，不叠加 LQE 或其他实验。

## 替换范围

| 层号（从 0 开始） | 位置 | YOLO11n 实际输入→输出通道 |
|---|---|---|
| 0 | Backbone，输入→P1/2 | 3→16 |
| 1 | Backbone，P1→P2/4 | 16→32 |
| 3 | Backbone，P2→P3/8 | 64→64 |
| 5 | Backbone，P3→P4/16 | 128→128 |
| 7 | Backbone，P4→P5/32 | 128→256 |
| 17 | Neck，P3→P4 | 64→64 |
| 20 | Neck，P4→P5 | 128→128 |

本机包来源：`mianju/ultralytics/cfg/models/11/yolo11-FSConv.yaml`、
`extra_modules/block.py::FSConv` 和 `extra_modules/attention.py::SEAttention`。
出处为 FSCFNet，TGRS 2025：<https://github.com/qzxwj/FSCFNet>。

保留本机包的分组卷积、空间分支、Haar 小波高频三子带融合、SEAttention、
低频卷积及最终拼接融合。保留 `DWTForward(J=1, mode='zero', wave='haar')` 后端，
没有改为近似小波实现。仅将类整理到独立模块，DWT 的导入放在构造函数内。
其余 C3k2、SPPF、C2PSA、原始 Detect、损失、TAL 和 NMS 均沿用基线。
这次是完整七处替换实验，结果不能用来区分七处中哪一处有效。

## 服务器训练

在独立工作目录执行：

```bash
python -m pip install -r requirements-fsconv.txt
WANDB_MODE=online python train.py
```

额外依赖为 `pytorch-wavelets==1.3.0`、`PyWavelets>=1.4.1`，以及为该版本
`pkg_resources` 导入保留的 `setuptools>=57,<81`。不需要安装 MMCV 或编译 CUDA 扩展。

使用混合 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、
seed=0、amp=False、workers=6、GPU1。从 YAML 随机初始化，与基线训练入口一致。
数据 YAML 保持指向服务器主目录 `/home/ot/Students/zx/yolo11-mixed-modality`，
无需复制 RGB/IR 数据或配对，模型输入仍为单张三通道图像。

训练输出：`falling-detection2/yolo11-real-rgb-ir-mixed-fsconv-full-640/weights/best.pt`。
同名目录存在时 Ultralytics 可能递增后缀，此时用验证脚本的 `--weights` 指定实际权重。
W&B 在线连接失败时，可改用 `WANDB_MODE=offline python train.py`。

## 验证与出表

训练完成后执行 `python val.py`，分别对原始基线及 FSConv 的 best.pt 评估：
RGB 用 `data_rgb.yaml`，真实 IR 用 `data.yaml`，统一 `split=val`、640、batch=4、
workers=0、GPU1。每种模态输出 mAP50、AP75、mAP50–95、Recall，保留四位小数；
变化为 `(FSConv−基线)×100` 个百分点。

表格保存到 `falling-detection2/fsconv-validation/comparison.md` 与 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 本地检查

```bash
python tests/check_fsconv.py --smoke-train
```

可加 `--reference-pack PATH`，直接与用户模块包对比 FSConv 的输出、输入及参数梯度，
并核对整个 YAML 除类别数外一致。其余检查包括独立 Haar 数值及梯度、七处位置与通道、
检测损失和 AdamW 更新、640/矩形推理、Conv/BN 融合、FP16 权重重载、
四项验证指标出表，以及合成数据 1 轮训练、EMA、best.pt 保存与重载验证。
合成数据检查只验证程序流程，不提供真实检测精度结论。

2026-11-24 本地以上检查全部通过（含本机包逐项对照）：Ultralytics 8.3.6、
Python 3.10.20、PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop GPU。
模型参数为 2,190,181 个。服务器的 200 轮训练和真实 RGB/IR 验证尚未执行。

## GitHub 离线备用

配套 `fsconv-full.bundle` 为基于 `4c9418e` 的增量包。上传服务器后，在主仓库执行：

```bash
git bundle verify /path/to/fsconv-full.bundle
git fetch /path/to/fsconv-full.bundle codex/experiment-fsconv-full:refs/remotes/offline/experiment-fsconv-full
git worktree add -b codex/experiment-fsconv-full \
  /home/ot/Students/zx/yolo11-mixed-modality-fsconv-full \
  refs/remotes/offline/experiment-fsconv-full
```
