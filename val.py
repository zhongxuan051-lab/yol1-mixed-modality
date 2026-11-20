"""Compare FreqFusion with the unchanged baseline on RGB and IR validation sets."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ot/Students/zx/yolo11-mixed-modality"))
    parser.add_argument("--weights", type=Path, default=root / "falling-detection2/yolo11-real-rgb-ir-mixed-freqfusion-full-640/weights/best.pt")
    parser.add_argument("--device", default="1")
    args = parser.parse_args()
    experiments = {
        "Baseline": args.dataset_root / "falling-detection2/yolo11-real-rgb-ir-mixed-640/weights/best.pt",
        "FreqFusion": args.weights,
    }
    for weights in experiments.values():
        if not weights.is_file():
            raise FileNotFoundError(weights)
    rows = []
    scores = {}
    for name, weights in experiments.items():
        for modality, data in (("RGB", "data_rgb.yaml"), ("IR", "data.yaml")):
            metrics = YOLO(str(weights)).val(
                data=str(args.dataset_root / data), split="val",
                imgsz=640, batch=4, workers=0, device=args.device, plots=False,
            ).box
            scores[name, modality] = metrics
            rows.append(
                f"{name} | {modality} | P {metrics.mp:.4f} | R {metrics.mr:.4f} | "
                f"mAP50 {metrics.map50:.4f} | AP75 {metrics.map75:.4f} | "
                f"mAP50-95 {metrics.map:.4f}"
            )
    print("\n" + "\n".join(rows), flush=True)
    print("\n| 验证集 | 指标 | 基线 | FreqFusion | 变化（百分点） |")
    print("|---|---|---:|---:|---:|")
    for modality in ("RGB", "IR"):
        baseline = scores["Baseline", modality]
        candidate = scores["FreqFusion", modality]
        for label, attr in (("AP75", "map75"), ("mAP50–95", "map"), ("Recall", "mr")):
            before, after = getattr(baseline, attr), getattr(candidate, attr)
            display = "真实 IR" if modality == "IR" else modality
            print(f"| {display} | {label} | {before:.4f} | {after:.4f} | {(after - before) * 100:+.2f} |")


if __name__ == "__main__":
    main()
