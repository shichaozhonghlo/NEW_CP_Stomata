import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('I:/ultralytics-yolo11-main/ultralytics/cfg/models/11/yolo11-C3k2-ESC-seg.yaml') # YOLO11
    model.load('yolo11n-seg.pt') # loading pretrain weights
    model.train(data='I:/05-全同胞气孔标记/ultralytics-yolo11-main/ultralytics/cfg/datasets/coco8-seg-new.yaml',
                imgsz=480,
                epochs=2000,
                batch=2,
                close_mosaic=0, 
                workers=8,  
                # device='0,1', 
                optimizer='SGD', 
                patience=100, 
                # resume=True, 
                # amp=False,
                # fraction=0.2,
                project='runs/train',
                name='exp',
                cache=True,
                amp=True,
                lr0=0.005,

               degrees=45,
               scale=0.8,
               hsv_h=0.3,
               hsv_s=0.6,
               hsv_v=0.3,
               copy_paste=0.5,
               erasing=0.5,
               mosaic=1.0,
               translate=0.05,
               shear=0.1                   
               )