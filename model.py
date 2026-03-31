from ultralytics import YOLO

class RebarDetectionModel:
    def __init__(self, model_path='yolov8n.pt'):
        # Load YOLOv8 model (nano version for speed)
        self.model = YOLO(model_path)

    def predict(self, image_path, conf=0.25, min_confidence=0.75):
        """
        Predict rebar exposure in image.
        Returns: dict with 'has_exposed_rebar' (bool) and 'confidence' (float).

        Args:
            image_path (str): Path to the image file.
            conf (float): YOLO confidence threshold for detections.
            min_confidence (float): Minimum confidence to count as positive rebar exposure.
        """
        results = self.model(image_path, conf=conf)

        max_conf = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls.item())
                    conf_val = box.conf.item()

                    # Assuming class 0 is 'rebar' or similar
                    # You may need to adjust based on your trained model
                    if cls == 0:
                        max_conf = max(max_conf, conf_val)

        has_rebar = max_conf >= min_confidence

        return {
            'has_exposed_rebar': has_rebar,
            'confidence': max_conf
        }

    def train(self, data_yaml, epochs=50, imgsz=640):
        """Train the model"""
        self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

    def export_onnx(self, output_path='model.onnx'):
        """Export model to ONNX for web deployment"""
        self.model.export(format='onnx', dynamic=True)
