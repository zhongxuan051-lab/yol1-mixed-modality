
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
import os
# os.environ["WANDB_MODE"] = "disabled"

def main():



    # 训练
    model = YOLO(r"ultralytics/cfg/models/11/yolo11.yaml")
    model.train(data=r"data.yaml",  # 数据集路径
                batch=4,
                imgsz=1920,
                epochs=200,
                amp=True,
                workers=2,
                optimizer='AdamW',
                lr0=0.001,
                device='0',
                project="falling-detection",  # W&B 项目名
                name="yolo11-ir1920"
                )


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()  # 对于Windows是必须的
    main()


