# DRFD 单处下采样实验

分支 `codex/experiment-drfd-p3`，独立起点为原始混合训练基线 `4c9418e`。
在 YOLO11n 的 Backbone 第 3 层（从 0 编号）用 DRFD 替换原步长为 2 的卷积。
这一层把 P2 特征降到 P3。在 640 方形输入下，尺寸为 160×160 → 80×80，
输入和输出均为 64 通道。采用本机包的 DRFD 和 Cut 实现，只使用现成 PyTorch 算子。
其余层沿用原始基线，不叠加 ADown 或 LaSEA。

DRFD 将空间到通道重排、步长卷积和最大池化三路的输出拼接，随后通过 1×1 卷积融合。
本实验检验在主干生成 P3 时采用这种下采样方式，是否能改善微小人员目标的召回。
来源为 RFD（TGRS 2023），<https://github.com/AeroVILab-AHU/RFD>。

在服务器独立工作区运行：

```bash
WANDB_MODE=online python train.py
```

沿用 `data_mixed.yaml`、单张三通道输入、640、batch=32、200 轮、AdamW、
lr0=0.001、seed=0、AMP 关闭、workers=6、GPU1，从 YAML 随机初始化。
权重输出 `falling-detection2/yolo11-real-rgb-ir-mixed-drfd-p3-640/weights/best.pt`。
如果训练目录增加了数字后缀，验证时使用 `--weights` 指定实际路径。

```bash
python val.py
```

在两个验证集上分别比较原始混合基线和 DRFD 的 best.pt。
RGB 使用 `data_rgb.yaml`，真实 IR 使用 `data.yaml`。两者统一使用 split=val、
640、batch=4、workers=0、GPU1。输出 `falling-detection2/drfd-p3-validation`
内的 `comparison.md` 和 `comparison.csv`，含 mAP50、AP75、mAP50–95、Recall。

`size_recall.md` 复用同一轮验证推理，按变换到 640 验证输入的真值框短边
划分 <8、8–<16、16–<32、≥32 px 四组。每张图按置信度降序，以 IoU≥0.50
进行一对一匹配。对每种模态，在两个模型上分别选择同一 FP 上限内能保留最多预测的
阈值（同分预测作为一组），再报告每组召回。RGB 和 IR 默认上限均为 216，
可分别用 `--fp-budget-rgb` 和 `--fp-budget-ir` 调整。表中列出两模型的实际
FP 和置信度阈值。候选预测总 FP 不足或分数并列跨越上限时，实际 FP 可低于上限。
该诊断的匹配方式独立于 Ultralytics 常规 AP 计算。召回基于现有标注。

GitHub 连接失败时，将增量包 `drfd-p3.bundle` 上传至服务器，在主仓库执行：

```bash
git bundle verify /path/to/drfd-p3.bundle
git fetch /path/to/drfd-p3.bundle codex/experiment-drfd-p3:refs/remotes/offline/experiment-drfd-p3
git worktree add -b codex/experiment-drfd-p3 \
  /home/ot/Students/zx/yolo11-mixed-modality-drfd-p3 \
  refs/remotes/offline/experiment-drfd-p3
```

进入新目录后执行训练命令。
