"""Compare FDConv with the unchanged baseline on RGB and IR validation sets."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/ot/Students/zx/yolo11-mixed-modality"))
    parser.add_argument("--weights", type=Path, default=root / "falling-detection2/yolo11-real-rgb-ir-mixed-fdconv-full-640/weights/best.pt")
    parser.add_argument("--device", default="1")
    args = parser.parse_args()
    experiments = {
        "Baseline": args.dataset_root / "falling-detection2/yolo11-real-rgb-ir-mixed-640/weights/best.pt",
        "FDConv": args.weights,
    }
    for weights in experiments.values():
        if not weights.is_file():
            raise FileNotFoundError(weights)
    rows = []
    for name, weights in experiments.items():
        for modality, data in (("RGB", "data_rgb.yaml"), ("IR", "data.yaml")):
            metrics = YOLO(str(weights)).val(
                data=str(args.dataset_root / data), split="val",
                imgsz=640, batch=4, workers=0, device=args.device, plots=False,
            ).box
            rows.append(
                f"{name} | {modality} | P {metrics.mp:.4f} | R {metrics.mr:.4f} | "
                f"mAP50 {metrics.map50:.4f} | AP75 {metrics.map75:.4f} | "
                f"mAP50-95 {metrics.map:.4f}"
            )
    print("\n" + "\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
