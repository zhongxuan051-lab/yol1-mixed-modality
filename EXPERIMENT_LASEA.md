# C3k2-LaSEA 全八处替换实验

分支：`codex/experiment-lasea-full`。共同起点：原始混合训练基线 `4c9418e`。
按本机包 `ultralytics/cfg/models/11/yolo11-C3k2-LaSEA.yaml` 的完整布局，
将全部八处 C3k2 换成 C3k2_LaSEA，类别数沿用基线 `nc=1`。

## 结构与机制

Backbone 层号为 **2、4、6、8**，Neck 层号为 **13、16、19、22**，均从 0 开始。
第 6、8、22 层启用嵌套 C3k，每处包含两个 LaSEA；其余每处一个，共 11 个 LaSEA。
保留原始下采样、SPPF、C2PSA、Detect、损失函数、TAL 和 NMS。
单张三通道输入，RGB 与真实 IR 混合训练，无需配对，不叠加此前模型改进。

LaSEA 使用膨胀率 1/2/3/4 的四路 3×3 卷积，拼接后通道混洗并融合，
通过池化注意力加权，最后残差相加。训练期从 1/2/3 三个池化尺度随机选择，
并使用空间打乱和随机采样；验证期固定全局平均池化，不使用随机采样。
随机性同时涉及 Python random 和 PyTorch，训练继续使用 Ultralytics 的 seed=0 设置。

来源：本机包 `extra_modules/LaSEA.py` 和 `extra_modules/block.py` 中的封装类。
原论文为 DCCS-Det：<https://arxiv.org/abs/2601.16428>。
这是模块移植实验，不是整个 DCCS-Det 的复现。

## 对原包的明确修正

原包 LaSEA 的五处卷积写作 `Conv(..., act=nn.ReLU)`。该项目及原包的 Conv
仅识别激活函数实例，因此此写法实际得到 `nn.Identity()`，没有执行 ReLU。
本实验将这五个构造调用修正为 `act=nn.ReLU()`，使四路卷积和融合卷积实际执行 ReLU。
未修改全局 Conv 行为，其他层的激活方式保留。

八处模型布局、默认通道、池化策略和其他算子保持包内设置。数值对照使用同步修正
这五个调用的包内参考实现，不能称为与原包未经修正的运行结果完全相同。

## 训练

在服务器独立工作目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、seed=0、
amp=False、workers=6、GPU1。从 YAML 随机初始化，与原基线训练入口一致。
数据 YAML 继续指向 `/home/ot/Students/zx/yolo11-mixed-modality`，无需复制数据。
本改动仅依赖现有 PyTorch，不需要 FSConv 的小波依赖或额外编译扩展。

权重输出：`falling-detection2/yolo11-real-rgb-ir-mixed-lasea-full-640/weights/best.pt`。
同名运行目录存在时 Ultralytics 可能递增后缀，验证时请用 `--weights` 指定实际权重。
如 W&B 在线连接失败，可改用 `WANDB_MODE=offline python train.py`。

## 验证与出表

训练完成执行 `python val.py`。分别评估原始混合基线与 LaSEA 的 best.pt：
RGB 使用 `data_rgb.yaml`，真实 IR 使用 `data.yaml`。统一 `split=val`、640、
batch=4、workers=0、GPU1；输出 mAP50、AP75、mAP50–95、Recall，四位小数。
变化为 `(LaSEA−基线)×100` 个百分点。

打印表格并保存 `falling-detection2/lasea-validation/comparison.md` 和 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 检查

```bash
python tests/check_lasea.py --smoke-train
```

可加 `--reference-pack PATH`，对照修正 ReLU 后的本机包：普通与嵌套 C3k2 路径、
训练和验证输出、输入与参数梯度以及整个 YAML。其余检查包括池化行为与可复现性、
八处位置/11 个内部模块、实际 ReLU 激活、检测损失、CUDA 反向传播和 AdamW、
640/矩形重复推理、Conv/BN 融合、FP16 权重重载、出表参数与格式，以及合成数据
一轮训练、EMA、best.pt 保存及重载验证。
合成数据检查不提供真实精度结论；服务器 200 轮训练与真实验证需后续运行。

2026-11-24 本地上述检查全部通过，环境为 Ultralytics 8.3.6、Python 3.10.20、
PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop GPU，参数量 4,267,587。
本地 CUDA 的 adaptive_avg_pool2d 反向传播提示无确定性实现，训练正常完成；
固定种子的池化采样检查不等于承诺整个 CUDA 训练逐位可复现。

## 离线 Git 备用

配套 `lasea-full.bundle` 是以 `4c9418e` 为前提的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/lasea-full.bundle
git fetch /path/to/lasea-full.bundle codex/experiment-lasea-full:refs/remotes/offline/experiment-lasea-full
git worktree add -b codex/experiment-lasea-full \
  /home/ot/Students/zx/yolo11-mixed-modality-lasea-full \
  refs/remotes/offline/experiment-lasea-full
```
