"""Compare baseline and DEConv best.pt on RGB and real IR validation sets."""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from size_recall import SIZE_LABELS, SizeRecallValidator, recall_at_budget


def nonnegative_int(value):
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("FP budget must be nonnegative")
    return count


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ot/Students/zx/yolo11-mixed-modality"))
    parser.add_argument("--weights", type=Path, default=root / "falling-detection2/yolo11-real-rgb-ir-mixed-deconv-reg-640/weights/best.pt")
    parser.add_argument("--baseline-weights", type=Path)
    parser.add_argument("--device", default="1")
    parser.add_argument("--fp-budget-rgb", type=nonnegative_int, default=216)
    parser.add_argument("--fp-budget-ir", type=nonnegative_int, default=216)
    parser.add_argument("--output-dir", type=Path, default=root / "falling-detection2/deconv-reg-validation")
    args = parser.parse_args()
    experiments = {
        "Baseline": args.baseline_weights or args.dataset_root / "falling-detection2/yolo11-real-rgb-ir-mixed-640/weights/best.pt",
        "DEConv": args.weights,
    }
    datasets = {"RGB": args.dataset_root / "data_rgb.yaml", "真实 IR": args.dataset_root / "data.yaml"}
    for path in (*experiments.values(), *datasets.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    scores, size_stats = {}, {}
    budgets = {"RGB": args.fp_budget_rgb, "真实 IR": args.fp_budget_ir}
    for name, weights in experiments.items():
        for modality, data in datasets.items():
            def make_validator(**kwargs):
                validator = SizeRecallValidator(**kwargs)
                size_stats[name, modality] = validator
                return validator

            metrics = YOLO(str(weights)).val(
                validator=make_validator, data=str(data), split="val", imgsz=640, batch=4, workers=0,
                conf=0.001, iou=0.7, max_det=300,
                device=args.device, plots=False,
                project=str(args.output_dir), name=f"{name.lower()}-{'rgb' if modality == 'RGB' else 'ir'}",
                exist_ok=True,
            ).box
            scores[name, modality] = {attr: float(getattr(metrics, attr)) for attr in ("map50", "map75", "map", "mr")}

    header = ["验证集", "指标", "基线", "DEConv", "变化（百分点）"]
    rows = []
    for modality in datasets:
        for label, attr in (("mAP50", "map50"), ("AP75", "map75"), ("mAP50–95", "map"), ("Recall", "mr")):
            before, after = scores["Baseline", modality][attr], scores["DEConv", modality][attr]
            rows.append([modality, label, f"{before:.4f}", f"{after:.4f}", f"{(after - before) * 100:+.2f}"])

    table = "\n".join([
        "| " + " | ".join(header) + " |",
        "|---|---|---:|---:|---:|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)
    print("\n" + table, flush=True)
    size_table = build_size_table(size_stats, budgets)
    (args.output_dir / "size_recall.md").write_text(size_table + "\n", encoding="utf-8")
    print("\n" + size_table, flush=True)
    print(f"\nSaved both comparisons to {args.output_dir}", flush=True)


def build_size_table(size_stats, budgets):
    lines = [
        "按 640 验证输入上的真值框短边分组，分数降序进行一对一 IoU≥0.50 匹配。",
        "",
        "| 验证集 | 短边（px） | 目标数 | 基线召回 | DEConv 召回 | 变化（百分点） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    operating_points = []
    for modality, budget in budgets.items():
        baseline = size_stats["Baseline", modality]
        candidate = size_stats["DEConv", modality]
        if (baseline.targets != candidate.targets).any():
            raise ValueError(f"{modality}: different target counts for the two models")
        base_tp, base_fp, base_conf = recall_at_budget(baseline.detections, budget)
        new_tp, new_fp, new_conf = recall_at_budget(candidate.detections, budget)
        for label, total, before, after in zip(SIZE_LABELS, baseline.targets, base_tp, new_tp):
            values = f"{before / total:.2%} | {after / total:.2%} | {(after - before) * 100 / total:+.2f}" if total else "— | — | —"
            lines.append(f"| {modality} | {label} | {total} | {values} |")
        base_cutoff = "无保留预测" if base_conf is None else f"置信度 ≥ {base_conf:.9g}"
        new_cutoff = "无保留预测" if new_conf is None else f"置信度 ≥ {new_conf:.9g}"
        operating_points.append(
            f"- {modality} FP 上限 {budget}。基线实际 FP {base_fp}（{base_cutoff}），"
            f"DEConv 实际 FP {new_fp}（{new_cutoff}）。"
        )
    return "\n".join(lines + ["", *operating_points])


if __name__ == "__main__":
    main()
