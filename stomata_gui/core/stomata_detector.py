from ultralytics import YOLO

class StomataDetector:

    def __init__(self, weight_path: str):
        self.weight_path = weight_path
        self.model = YOLO(weight_path)

    def switch_model(self, new_weight: str):
        self.weight_path = new_weight
        self.model = YOLO(new_weight)

    def predict_image(self, img_path, conf=0.25, iou=0.45):
        results = self.model(
            img_path,
            conf=conf,
            iou=iou
        )
        return results