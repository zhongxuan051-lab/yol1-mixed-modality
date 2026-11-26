# C2PSA-MALA 单层注意力实验

分支：`codex/experiment-mala`。共同起点：原始混合训练基线 `4c9418e`。
按本机包 `ultralytics/cfg/models/11/yolo11-C2PSA-MALA.yaml`，仅将第 10 层
C2PSA 换成 C2PSA_MALA，类别数保持 `nc=1`。这是从原始基线出发的独立实验。

## 改动与机制

第 10 层的输入/输出均为 256 通道，C2PSA 内部处理分支为 128 通道，MALA
使用 2 个注意力头。保留 C2PSA 的投影卷积、分流结构、前馈网络与残差设置，
只替换 PSABlock 中的注意力算子。其他层、Neck、Detect、损失、TAL、NMS
与原始基线相同。单图三通道输入，RGB 与真实 IR 混合训练无需配对。

MALA 来自 ICCV 2025《Rectifying Magnitude Neglect in Linear Attention》：
<https://arxiv.org/abs/2507.00698>，官方项目：<https://github.com/qhfan/MALA>。
本实验移植用户包的 MALA 与 C2PSA 封装，不是完整 MAViT 主干复现。
原始 C2PSA 使用 Softmax Attention，因此本轮是在比较两种注意力设计，
不能直接认定基线存在论文针对的线性注意力幅值缺失问题。

保留包内正值 Q/K 映射、幅值相关归一化与修正、二维 RoPE、5×5 深度卷积、
输出门控及投影。包中的两处 `einops.rearrange` 用等价的 PyTorch
reshape/permute/transpose 实现，输出与梯度通过原包对照；不引入额外运行依赖。
保留包中的位置编码缓冲区、`1e-6` 稳定项及其他默认参数。

## 训练

在服务器独立工作目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、seed=0、
amp=False、workers=6、GPU1。从 YAML 随机初始化，与原基线训练入口一致。
数据配置仍指向 `/home/ot/Students/zx/yolo11-mixed-modality`，无需复制数据。
训练及推理仅使用现有 PyTorch，无需安装 einops 或编译扩展。
若 W&B 在线初始化失败，可用 `WANDB_MODE=offline python train.py`。

训练输出：`falling-detection2/yolo11-real-rgb-ir-mixed-mala-640/weights/best.pt`。
同名目录存在时 Ultralytics 可能递增后缀，验证用 `--weights` 指定实际权重。

## 验证与出表

训练完成后执行 `python val.py`，分别评估原始混合基线与 MALA 的 best.pt。
RGB 用 `data_rgb.yaml`，真实 IR 用 `data.yaml`。统一 `split=val`、640、
batch=4、workers=0、GPU1，报告 mAP50、AP75、mAP50–95、Recall，四位小数。
变化为 `(MALA−基线)×100` 个百分点。

表格保存到 `falling-detection2/mala-validation/comparison.md` 与 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 检查

```bash
python tests/check_mala.py --smoke-train
```

测试以显式 token×token 权重矩阵作为独立参考，对比线性重排计算的输出与输入、
参数梯度，并检查矩形二维 RoPE、零输入反向传播。集成检查覆盖单层布局、保留的
C2PSA 投影及前馈网络、CUDA 检测损失和 AdamW、640/矩形及 batch=1/4 推理、
Conv/BN 融合、FP16 权重重载、出表参数/格式，以及合成数据一轮训练、EMA、
best.pt 保存和重载验证。

可追加 `--reference-pack PATH`，直接核对本机包原始 einops 写法、完整 C2PSA
的训练/验证输出及全部梯度、整个 YAML；只有该可选原包对照需要本地安装 einops。
合成数据仅检查程序流程，真实精度须由服务器训练和 RGB/IR 验证确认。

2026-11-26 上述本地检查全部通过（含原包对照），环境为 Ultralytics 8.3.6、
Python 3.10.20、PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop GPU。
模型共 2,624,595 个参数，较当前单类基线增加 34,560 个。
服务器 200 轮训练与真实验证尚未执行。

## 离线 Git 备用

配套 `mala.bundle` 为基于 `4c9418e` 的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/mala.bundle
git fetch /path/to/mala.bundle codex/experiment-mala:refs/remotes/offline/experiment-mala
git worktree add -b codex/experiment-mala \
  /home/ot/Students/zx/yolo11-mixed-modality-mala \
  refs/remotes/offline/experiment-mala
```
