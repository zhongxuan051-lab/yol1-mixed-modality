# C3k2-DRG 全八处替换实验

分支：`codex/experiment-drg-full`。共同起点：原始混合训练基线 `4c9418e`。
采用本机包 `ultralytics/cfg/models/11/yolo11-C3k2-DRG.yaml` 的完整八处布局，
类别数沿用基线 `nc=1`。这是独立实验，不与候选 LaSEA 叠加。

## 改动范围

将 Backbone 第 **2、4、6、8** 层和 Neck 第 **13、16、19、22** 层的 C3k2
替换为 C3k2_DRG，层号从 0 开始。第 6、8、22 层保留嵌套 C3k_DRG，
各包含两个 DRG，其余每处一个，YOLO11n 共 11 个 DRG。
每个 DRG 按包内默认包含一个 RCSAB，使用 3×3 卷积。
其余下采样、SPPF、C2PSA、Detect、损失、TAL 和 NMS 沿用原始基线。

来源：用户本机包 `extra_modules/block.py` 的 DRPCA-Net 区域，包含
ChannelAttention、DynamicSpatialAttention、RCSAB、DRG 及 C3k/C3k2 封装。
原论文：DRPCA-Net，TGRS，<https://arxiv.org/abs/2507.09541>。
这是 DRG 模块移植实验，不是完整 RPCA 分解或动态展开网络的复现。

## 机制

RCSAB 使用残差卷积、通道注意力和动态空间注意力。通道注意力使用全局平均与
最大池化；空间注意力对每张图的当前特征进行池化，生成该样本的 3×3 卷积核，
作用于通道平均特征，经过 sigmoid 后生成空间权重。
使用分组卷积批量计算时，每张图的卷积核只作用于该图，不在样本之间混合。
DRG 再使用末端卷积和长残差连接整合特征。

保留包中的 BN、LeakyReLU、卷积偏置、注意力与残差设置。
训练和验证均使用同一个动态空间注意力机制；不引入 LaSEA 的随机池化策略。
不额外添加质量估计分支或损失函数。

## 服务器训练

在独立工作目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、seed=0、
amp=False、workers=6、GPU1。从 YAML 随机初始化，与基线训练入口一致。
数据配置仍指向 `/home/ot/Students/zx/yolo11-mixed-modality`，无需复制数据。
RGB 与真实 IR 不需配对，模型仍每次输入一张三通道图像。
模块只使用现有 PyTorch 算子，不需要额外依赖或编译 CUDA 扩展。

训练输出：`falling-detection2/yolo11-real-rgb-ir-mixed-drg-full-640/weights/best.pt`。
同名目录存在时 Ultralytics 可能增加后缀，此时验证用 `--weights` 指定实际权重。
如 W&B 在线初始化失败，可使用 `WANDB_MODE=offline python train.py`。

## 验证

训练完成执行 `python val.py`，分别对原始混合基线与 DRG 的 best.pt 评估：
RGB 使用 `data_rgb.yaml`，真实 IR 使用 `data.yaml`。统一 `split=val`、640、
batch=4、workers=0、GPU1。输出 mAP50、AP75、mAP50–95、Recall，四位小数。
变化为 `(DRG−基线)×100` 个百分点，表头为“验证集｜指标｜基线｜C3k2-DRG｜变化（百分点）”。

打印并保存 `falling-detection2/drg-validation/comparison.md` 与 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 本地检查

```bash
python tests/check_drg.py --smoke-train
```

可加 `--reference-pack PATH`，与本机包对照普通/嵌套 C3k2 的训练及验证输出、
输入与参数梯度，并核对整个 YAML。另以逐图单独卷积作为独立参考，验证批量动态卷积
的数值、梯度与样本隔离。集成检查覆盖八处位置/11 个内部 DRG、检测损失、CUDA
反向传播、AdamW、640/矩形及 batch=1/4 推理、Conv/BN 融合、FP16 权重重载、
出表参数与格式，以及合成数据一轮训练、EMA、best.pt 保存和重载验证。
合成数据仅验证程序流程，不提供真实精度结论；服务器 200 轮训练与真实评估需后续执行。

2026-11-24 上述本地检查全部通过（含本机包对照）：Ultralytics 8.3.6、
Python 3.10.20、PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop GPU，
模型参数量 2,985,127。本地 adaptive_max_pool2d 的 CUDA 反向传播提示无确定性
实现，训练正常完成；固定 seed 不等于承诺整个 CUDA 训练逐位可复现。

## 离线 Git 备用

配套 `drg-full.bundle` 是基于 `4c9418e` 的增量包。上传服务器后，在主仓库执行：

```bash
git bundle verify /path/to/drg-full.bundle
git fetch /path/to/drg-full.bundle codex/experiment-drg-full:refs/remotes/offline/experiment-drg-full
git worktree add -b codex/experiment-drg-full \
  /home/ot/Students/zx/yolo11-mixed-modality-drg-full \
  refs/remotes/offline/experiment-drg-full
```
