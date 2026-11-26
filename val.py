"""Compare baseline and ADown best.pt on RGB and real IR validation sets."""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ot/Students/zx/yolo11-mixed-modality"))
    parser.add_argument("--weights", type=Path, default=root / "falling-detection2/yolo11-real-rgb-ir-mixed-adown-640/weights/best.pt")
    parser.add_argument("--baseline-weights", type=Path)
    parser.add_argument("--device", default="1")
    parser.add_argument("--output-dir", type=Path, default=root / "falling-detection2/adown-validation")
    args = parser.parse_args()
    experiments = {
        "Baseline": args.baseline_weights or args.dataset_root / "falling-detection2/yolo11-real-rgb-ir-mixed-640/weights/best.pt",
        "ADown": args.weights,
    }
    datasets = {"RGB": args.dataset_root / "data_rgb.yaml", "真实 IR": args.dataset_root / "data.yaml"}
    for path in (*experiments.values(), *datasets.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    scores = {}
    for name, weights in experiments.items():
        for modality, data in datasets.items():
            metrics = YOLO(str(weights)).val(
                data=str(data), split="val", imgsz=640, batch=4, workers=0,
                device=args.device, plots=False,
                project=str(args.output_dir), name=f"{name.lower()}-{'rgb' if modality == 'RGB' else 'ir'}",
                exist_ok=True,
            ).box
            scores[name, modality] = {attr: float(getattr(metrics, attr)) for attr in ("map50", "map75", "map", "mr")}

    header = ["验证集", "指标", "基线", "ADown", "变化（百分点）"]
    rows = []
    for modality in datasets:
        for label, attr in (("mAP50", "map50"), ("AP75", "map75"), ("mAP50–95", "map"), ("Recall", "mr")):
            before, after = scores["Baseline", modality][attr], scores["ADown", modality][attr]
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
    print(f"\nSaved comparison.md and comparison.csv to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
