from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent

    rgb_weights = Path(
        "/home/ot/Students/zx/multi-detection/falling-detection2/"
        "yolo11-human detection.v2i.yolov11 -去时间戳rgb/"
        "weights/best.pt"
    )

    ir_weights = (
        root / "falling-detection2"
        / "yolo11-real-ir-640"
        / "weights" / "best.pt"
    )

    experiments = [
        (
            "RGB-only -> Real IR",
            rgb_weights,
            root / "data.yaml",       # 真实红外数据配置
        ),
        (
            "IR-only -> RGB",
            ir_weights,
            root / "data_rgb.yaml",   # 可见光数据配置
        ),
    ]

    for name, weights, data_yaml in experiments:
        assert weights.is_file(), f"找不到权重：{weights}"
        assert data_yaml.is_file(), f"找不到数据配置：{data_yaml}"

        print(f"\n===== {name} =====", flush=True)

        metrics = YOLO(str(weights)).val(
            data=str(data_yaml),
            split="test",
            imgsz=640,
            batch=4,
            workers=0,
            device=0,
            plots=False,
        )

        print(f"\n===== {name}: precise results =====", flush=True)
        print(f"Precision: {metrics.box.mp:.4f}", flush=True)
        print(f"Recall:    {metrics.box.mr:.4f}", flush=True)
        print(f"mAP50:     {metrics.box.map50:.4f}", flush=True)
        print(f"mAP50-95:  {metrics.box.map:.4f}", flush=True)


if __name__ == "__main__":
    main()
