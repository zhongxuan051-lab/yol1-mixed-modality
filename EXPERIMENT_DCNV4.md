# C3k2-DCNv4：按本机包默认配置

分支 `codex/experiment-dcnv4`，独立起点为原始混合训练基线 `4c9418e`。
采用本机包 `ultralytics/cfg/models/11/yolo11-C3k2-DCNV4.yaml`，仅将类别数
改为 `nc=1`，以 `yolo11n-dcnv4.yaml` 固定使用 nano 配置。

## 结构

只替换 Backbone 第 6、8 层（从 0 编号）的 C3k2；其余 Backbone、C2PSA、
Neck、P3/P4/P5 Detect、损失、TAL、NMS 沿用原始基线。
两处保留 `c3k=True`，nano 深度缩放后各含 2 个 Bottleneck，替换其第二层卷积，
合计 4 个 DCNv4。第 6 层内部两个算子各 32 通道，第 8 层各 64 通道。

遵循包内封装实际传参：`group=1`、3×3、stride=1、pad=1、dilation=1、
offset_scale=1、dw_kernel_size=None、remove_center=False、center_feature_scale=False，
保留 value/output 线性投影及偏置、零初始化 offset/mask、外层 BN 与 SiLU。
**不将包内 group=1 改成原始 DCNv4 类构造器默认的 group=4。**
offset/mask 权重不额外归一化，也不改成其他可变形卷积。

模块来自本机包 `extra_modules/block.py`，NCHW DCNv4 类来自
`extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py`。CUDA 源码随分支放在
`third_party/DCNv4_op`，仅清理尾随空白与行尾，保留原计算及版权头。论文为 CVPR 2024：
<https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_Efficient_Deformable_ConvNets_Rethinking_Dynamic_and_Sparse_Operator_for_Vision_CVPR_2024_paper.html>；
原仓库 <https://github.com/OpenGVLab/DCNv4>。

## CPU 初始化与 CUDA 训练

Ultralytics 8.3.6 构建检测模型时会在 CPU 上用虚拟图像计算步长。
包内扩展只有 CUDA 实现，因此为 CPU 提供按相同采样公式计算的 bilinear 参考路径。
该路径使用每组 K 对 (x,y) 偏移、K 个有符号权重，处理末尾对齐填充。
这项改动用于初始化与 CPU 检查，不改变模型配置、可训练参数或 GPU 算法。

**所有 GPU 前向/反向都强制使用包内编译扩展**，没有自动 CUDA 替代实现。
`train.py` 在训练开始前检查扩展是否可导入。缺少扩展或 ABI 不兼容时直接报错，
不会启动 200 轮后才发现初始化问题。扩展导入成功不等于数值检查通过，仍应执行下方自检。

## 服务器编译、自检、训练

在服务器训练用 Python 环境、独立实验目录执行：

```bash
python scripts/install_dcnv4.py &&
python tests/check_dcnv4.py --smoke-train &&
WANDB_MODE=online python train.py
```

编译需要可见的 CUDA GPU、CUDA toolkit（含 nvcc）、与当前 PyTorch 匹配的
CUDA 编译环境和 C++ 编译器。脚本打印 Python/PyTorch/CUDA_HOME/编译器路径，
用当前 Python 对本地源码执行 `pip install --no-build-isolation --no-deps`。
默认 MAX_JOBS=4，可通过环境变量调整。不安装或替换服务器 PyTorch、CUDA、驱动。
不需要 MMCV；不要跨 Python/PyTorch/CUDA 环境复制二进制扩展。

自检默认 GPU1（支持 `--device 0` 等），不上传 W&B；使用临时合成数据，
128、batch=2、1 轮，检查结束删除临时数据。自检的 CPU 参考与编译 CUDA 核心
会分别核对前向及输入、偏移、权重梯度。只有前两条命令成功才启动正式训练。

正式训练：混合 `data_mixed.yaml`、单张三通道图像、640、batch=32、200 轮、
AdamW、lr0=0.001、seed=0、amp=False、workers=6、GPU1，沿用基线 YAML 随机初始化。
RGB/真实 IR 无需配对，不叠加 LaSEA 或其他实验结构。
数据 YAML 继续指向服务器主目录，不需要复制数据。
W&B 初始化连接失败时可用 `WANDB_MODE=offline python train.py`。

权重：`falling-detection2/yolo11-real-rgb-ir-mixed-dcnv4-640/weights/best.pt`。
同名运行目录存在时 Ultralytics 可能自动递增后缀，验证请用 `--weights` 指定实际文件。

## 验证与表格

```bash
python val.py
```

分别用 baseline 与本实验 best.pt 在 `data_rgb.yaml`、`data.yaml` 上评估。
统一 split=val、640、batch=4、workers=0、GPU1、默认 FP32，与既有实验一致。
输出 RGB/真实 IR 的 mAP50、AP75、mAP50–95、Recall，四位小数，变化为
`(DCNv4−基线)×100` 个百分点，保留两位小数。
打印并保存 `falling-detection2/dcnv4-validation/comparison.md` 与 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 本地已完成检查与限制

2026-11-26，Python 3.10.20、Ultralytics 8.3.6、PyTorch 2.10.0.dev20251213+cu128：

```bash
python tests/check_dcnv4.py --reference-only --reference-pack /path/to/mianju --smoke-train
```

已通过：完整包配置一致（仅 nc=1）、DCNv4 类 AST 一致、包内 C3k2 封装训练/验证
输出及所有梯度一致、34 个 vendor 文件除空白格式外与原包一致；CPU 核心与 torchvision 独立
deform_conv2d 的输出/梯度一致，覆盖多组偏移、有符号非归一化权重、边界及填充；
4 算子布局、步长、两步检测损失/反向/AdamW、640 和矩形 batch=1/4 推理、
Conv/BN 融合、FP16 权重保存与 FP32 重载、验证调用及表格，以及合成数据完整一轮
训练、EMA、best.pt 保存重载与验证。

模型参数量 2,525,011（单类原始基线 2,590,035，减少 65,024）。
参数量减少不能推断实际速度；普通 FLOPs 统计不能完整计入此 CUDA 自定义算子。

**本机没有 nvcc 和 C++ 编译器，未编译或运行 CUDA 扩展。**
`--reference-only` 检查结果不能替代服务器上不带该选项的编译 CUDA 自检。
服务器真实 200 轮训练、RGB/IR 精度评估均尚未执行。

## Git 离线备用

配套 `dcnv4.bundle` 是基于 `4c9418e` 的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/dcnv4.bundle
git fetch /path/to/dcnv4.bundle codex/experiment-dcnv4:refs/remotes/offline/experiment-dcnv4
git worktree add -b codex/experiment-dcnv4 \
  /home/ot/Students/zx/yolo11-mixed-modality-dcnv4 \
  refs/remotes/offline/experiment-dcnv4
```

进入新目录后仍先编译扩展并自检，再开始正式训练。
