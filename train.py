from multiprocessing import freeze_support

from ultralytics import YOLO


def main():
    model = YOLO("ultralytics/cfg/models/11/yolo11n-p2-p3-detail.yaml")

    model.train(
        data="data_mixed.yaml",
        batch=32,
        imgsz=640,
        epochs=200,
        amp=False,
        workers=6,
        optimizer="AdamW",
        lr0=0.001,
        seed=0,
        device="1",
        project="falling-detection2",
        name="yolo11-real-rgb-ir-mixed-p2-p3-detail-640",
    )


if __name__ == "__main__":
    freeze_support()
    main()
