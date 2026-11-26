# C2PSA-CGLU 单层前馈网络实验

分支：`codex/experiment-cglu`。共同起点：原始混合训练基线 `4c9418e`。
采用本机包 `ultralytics/cfg/models/11/yolo11-C2PSA-CGLU.yaml`，仅将第 10 层
C2PSA 换成 C2PSA_CGLU，类别数保持 `nc=1`。本轮从原始基线独立出发。

## 结构与实现约定

第 10 层输入/输出均为 256 通道，内部处理分支为 128 通道。
保留原 C2PSA 的投影、分流、Softmax Attention 和外层残差，仅将 PSABlock
中的前馈网络换成 Convolutional GLU。其余层、Neck、Detect、损失、TAL 和 NMS
沿用基线。RGB 与真实 IR 混合训练无需配对，每次仍输入单张三通道图像。

来源：用户本机包 `extra_modules/transformer.py` 中的 ConvolutionalGLU、
PSABlock_CGLU、C2PSA_CGLU。原工作 TransNeXt（CVPR 2024）：
<https://arxiv.org/abs/2311.17132>，<https://github.com/DaiShiResearch/TransNeXt>。

计算为 `u,v = fc1(x).chunk(2)`，经 `GELU(DWConv3x3(u)) * v` 门控，
再由 fc2 投影输出。沿用包内 `hidden=int(2*(2*C)/3)`，C=128 时 hidden=170，
fc1 输出 340 通道、fc2 输出 128 通道，dropout=0，卷积偏置保留。

按本轮已确定的实验定义，**只保留外层一次残差**：

- 本机包 GLU 内部返回 `x+F(x)`，PSABlock 外层再相加，形成 `2*x+F(x)`。
- 本实验 GLU 仅返回 `F(x)`，外层保持 `x+F(x)`。

此项差异已明确记录，不宣称与未经调整的包内模块输出完全一致。
门控公式与包内 TransNeXt_native.py 中 token/Linear 版本对照，1×1 卷积与
Linear 权重对应后检查输出及所有梯度。

## 训练

在服务器独立工作目录执行：

```bash
WANDB_MODE=online python train.py
```

沿用混合 `data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、
seed=0、amp=False、workers=6、GPU1。从 YAML 随机初始化，与基线入口一致。
数据 YAML 继续指向 `/home/ot/Students/zx/yolo11-mixed-modality`，无需复制数据。
仅使用现有 PyTorch 算子，无需额外依赖或编译扩展。
如 W&B 在线初始化失败，可用 `WANDB_MODE=offline python train.py`。

训练输出：`falling-detection2/yolo11-real-rgb-ir-mixed-cglu-640/weights/best.pt`。
同名目录存在时 Ultralytics 可能递增后缀，验证用 `--weights` 指定实际路径。

## 验证与出表

训练后执行 `python val.py`，分别评估原始混合基线与 CGLU 的 best.pt。
RGB 用 `data_rgb.yaml`，真实 IR 用 `data.yaml`。统一 `split=val`、640、
batch=4、workers=0、GPU1，报告 mAP50、AP75、mAP50–95、Recall，四位小数。
变化为 `(CGLU−基线)×100` 个百分点。

打印并保存 `falling-detection2/cglu-validation/comparison.md` 和 `comparison.csv`。
支持 `--dataset-root`、`--weights`、`--baseline-weights`、`--device`、`--output-dir`。

## 检查

```bash
python tests/check_cglu.py --smoke-train
```

可追加 `--reference-pack PATH`，核对 token/Linear 版本及同步去除内部残差后的
包内 C2PSA-CGLU：训练/验证输出、输入及所有参数梯度、整个 YAML。
其他检查覆盖零变换时仅返回一次 x 及恒等梯度、单层布局与保留的原注意力、
CUDA 检测损失/AdamW、640/矩形及 batch=1/4 推理、Conv/BN 融合、FP16 保存重载、
出表参数与格式，以及合成数据一轮训练、EMA、best.pt 保存及重载验证。
合成数据仅验证程序流程；真实精度需在服务器训练后评估。

2026-11-26 上述本地检查全部通过（含原包及 token 参考对照），环境为
Ultralytics 8.3.6、Python 3.10.20、PyTorch 2.10.0.dev20251213+cu128、
RTX 5060 Laptop GPU。模型参数量 2,591,179，比当前单类基线增加 1,144 个。
服务器 200 轮训练和真实 RGB/IR 验证尚未执行。

## 离线 Git 备用

配套 `cglu.bundle` 为基于 `4c9418e` 的增量包。上传服务器后在主仓库执行：

```bash
git bundle verify /path/to/cglu.bundle
git fetch /path/to/cglu.bundle codex/experiment-cglu:refs/remotes/offline/experiment-cglu
git worktree add -b codex/experiment-cglu \
  /home/ot/Students/zx/yolo11-mixed-modality-cglu \
  refs/remotes/offline/experiment-cglu
```
