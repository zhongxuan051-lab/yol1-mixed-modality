# YOLO11n 混合训练：LQE 检测头

本实验从 `4c9418e` 出发，仅将层 23 的 `Detect` 换成 `Detect_LQE`，输入仍为
层 16、19、22 的 P3/P4/P5 特征。Backbone、Neck、DFL 解码、原有检测损失、
TAL 分配算法和 NMS 保留。RGB 与真实 IR 不需要配对，每次输入一张三通道图。

## 机制与来源

检验假设：利用回归分布统计修正分类分数，能否改善定位质量与检测分数的匹配，
从而提高 AP75 和 mAP50–95，同时检查 Recall 是否下降。

实现参考用户模块包中的 `Detect_LQE`（包内注明 GFLV2 来源）：
<https://arxiv.org/abs/2011.12885>。这是包内加性 LQE 的 YOLO11 适配实验，
不是完整 GFLV2 复现，不使用其整套损失，也不声称该模块为原创方法。

每个尺度独立使用四条边各 16 个 bin 的 softmax 分布，取每边 top-4 概率与
这四个概率的均值，得到 20 通道统计特征。经 `1×1 Conv(20→64) → ReLU →
1×1 Conv(64→1)` 预测修正值，加到原分类 logits，再走原生 sigmoid。
最后一层卷积权重与偏置初始化为零。回归统计不 detach，允许分类损失通过质量分支
反向传播到回归分支；其训练动力学也可能改变，不能把结果完全解释为推理重排序。

修正了模块包的分组顺序：YOLO DFL 连续排列的是 `[4, reg_max]`，因此在
`[B,4,16,H,W]` 的第 2 维执行 softmax/top-k，不能直接 reshape 为 `[B,16,4,H,W]`。
不引入 LSCD 或额外质量监督损失。

## 训练

在服务器新工作目录执行 `python train.py`。配置沿用混合基线：
`data_mixed.yaml`、640、batch=32、200 轮、AdamW、lr0=0.001、seed=0、
amp=False、workers=6、GPU1。从 YAML 随机初始化，与基线入口相同，不加载基线 best.pt 微调。
数据 YAML 中的数据路径仍指向 `/home/ot/Students/zx/yolo11-mixed-modality`。

权重输出：`falling-detection2/yolo11-real-rgb-ir-mixed-lqe-640/weights/best.pt`。
若同名运行目录已存在，Ultralytics 可能自动增加后缀；此时验证需用 `--weights` 指定实际路径。
如 W&B 在线初始化失败，可使用 `WANDB_MODE=offline python train.py`。

## 验证

训练完成后执行 `python val.py`。分别重评原始混合基线和 LQE 的 best.pt：
RGB 用 `data_rgb.yaml`，真实 IR 用 `data.yaml`，统一 `split=val`、640、batch=4、
workers=0、GPU1。测试集不参与本轮选择。

打印并保存“验证集｜指标｜基线｜LQE｜变化（百分点）”，按 mAP50、AP75、
mAP50–95、Recall 排列，每项四位小数，变化为 `(LQE−基线)×100`。
结果保存到 `falling-detection2/lqe-validation/comparison.md` 和 `comparison.csv`。

可选参数：`--weights PATH`、`--baseline-weights PATH`、`--dataset-root PATH`、
`--device 1`、`--output-dir PATH`。

## 本地检查

`python tests/check_lqe.py` 检查按边分组的统计数值、跨边隔离、初始输出与基线
一致性、CUDA 检测损失和梯度、AdamW 更新、640 与矩形推理、融合及权重重载。

`python tests/check_lqe.py --smoke-train` 额外以合成数据运行 1 轮训练、EMA、
best.pt 保存与重载验证。合成数据只用于验证程序流程，不提供真实检测精度结论。

2026-11-23 本地上述全部检查通过：Ultralytics 8.3.6、Python 3.10.20、
PyTorch 2.10.0.dev20251213+cu128、RTX 5060 Laptop GPU。LQE 模型共 2,594,262
参数，较本仓库单类基线增加 4,227 个。完整训练和真实验证集评估需在服务器运行；
本地检查不代表服务器环境已验证，也不代表检测精度已提升。

## 离线传输

若 GitHub HTTPS 不通，将配套 `lqe-head.bundle` 上传到服务器。该增量包以
`4c9418e` 为前提，适用于已有此基线提交的主仓库。在服务器主仓库执行：

```bash
git bundle verify /path/to/lqe-head.bundle
git fetch /path/to/lqe-head.bundle codex/experiment-lqe-head:refs/remotes/offline/experiment-lqe-head
git worktree add -b codex/experiment-lqe-head \
  /home/ot/Students/zx/yolo11-mixed-modality-lqe-head \
  refs/remotes/offline/experiment-lqe-head
```
