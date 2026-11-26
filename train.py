from multiprocessing import freeze_support

from ultralytics import YOLO
from ultralytics.nn.modules.dcnv4_backend import require_cuda_extension


def main():
    require_cuda_extension()
    model = YOLO("ultralytics/cfg/models/11/yolo11n-dcnv4.yaml")

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
        name="yolo11-real-rgb-ir-mixed-dcnv4-640",
        
    )


if __name__ == "__main__":
    freeze_support()
    main()
