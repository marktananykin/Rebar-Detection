from model import RebarDetectionModel
import requests
import base64
from io import BytesIO

class RebarDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = RebarDetectionModel(model_path)

    def predict(self, image_path, min_confidence=0.75):
        """
        Predict if an image contains exposed rebar using YOLOv8.

        Will iterate through descending YOLO confidence thresholds until the prediction reaches min_confidence.

        Args:
            image_path (str): Path to the image file.
            min_confidence (float): Minimum confidence for a positive detection.

        Returns:
            dict: Prediction result with class, confidence, and source.
        """
        best_confidence = 0.0
        best_result = {
            'prediction': 'No Exposed Rebar',
            'confidence': 0.0,
            'source': 'local',
            'status': 'low_confidence'
        }

        for threshold in [0.5, 0.4, 0.3, 0.2, 0.1]:
            result = self.model.predict(image_path, conf=threshold, min_confidence=min_confidence)
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']

            if result['has_exposed_rebar']:
                return {
                    'prediction': 'Exposed Rebar',
                    'confidence': result['confidence'],
                    'source': 'local',
                    'status': 'confident'
                }

        # No positive detection reached min_confidence.
        return {
            'prediction': 'No Exposed Rebar',
            'confidence': best_confidence,
            'source': 'local',
            'status': 'confidence_below_threshold' if best_confidence < min_confidence else 'confident'
        }


class RoboflowRebarDetector:
    def __init__(self, api_key, workspace='marks-workspace-dymtv', workflow='general-segmentation-api', classes='Exposed rebar', min_confidence=0.75):
        self.api_key = api_key
        self.workspace = workspace
        self.workflow = workflow
        self.classes = classes
        self.min_confidence = min_confidence
        self.api_url = f"https://serverless.roboflow.com/{workspace}/{workflow}"

    def predict(self, image_path):
        """
        Use Roboflow workflow to detect exposed rebar.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction result with class and confidence.
        """
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()

        files = {'file': ('image.jpg', BytesIO(image_data), 'image/jpeg')}
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.post(self.api_url, files=files, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Roboflow API error: {response.status_code} - {response.text}")

        result = response.json()
        max_confidence = 0.0

        if result and 'result' in result:
            workflow_result = result['result']

            if 'predictions' in workflow_result:
                for pred in workflow_result.get('predictions', []):
                    label = pred.get('class', '').lower()
                    if 'rebar' in label or 'exposed' in label:
                        max_confidence = max(max_confidence, pred.get('confidence', 0.0))

            elif 'segmentation' in workflow_result:
                for seg in workflow_result.get('segmentation', []):
                    label = seg.get('class', '').lower()
                    if 'rebar' in label or 'exposed' in label:
                        max_confidence = max(max_confidence, seg.get('confidence', 0.0))

            elif 'classes' in workflow_result:
                for cls in workflow_result.get('classes', []):
                    label = cls.get('name', '').lower()
                    if 'rebar' in label or 'exposed' in label:
                        max_confidence = max(max_confidence, cls.get('confidence', 0.0))

        if not max_confidence and self.classes.lower() in result:
            max_confidence = result.get(self.classes.lower(), {}).get('confidence', 0.0)

        has_exposed_rebar = max_confidence >= self.min_confidence
        prediction = 'Exposed Rebar' if has_exposed_rebar else 'No Exposed Rebar'

        return {
            'prediction': prediction,
            'confidence': max(0.0, min(1.0, max_confidence)),
            'source': 'roboflow',
            'status': 'confident' if has_exposed_rebar else 'confidence_below_threshold'
        }

def detect_rebar(image_path, model_path='rebar_model.pth', device='cpu', roboflow_api_key=None, min_confidence=0.75):
    """
    Convenience function to detect rebar in an image.

    Args:
        image_path (str): Path to the image.
        model_path (str): Path to the trained model (for local detection).
        device (str): Device to run inference on.
        roboflow_api_key (str): Optional Roboflow API key for enhanced detection.
        min_confidence (float): Minimum confidence to accept detected rebar.

    Returns:
        dict: Prediction result.
    """
    if roboflow_api_key:
        try:
            detector = RoboflowRebarDetector(roboflow_api_key, min_confidence=min_confidence)
            result = detector.predict(image_path)
            result['confidence_threshold'] = min_confidence
            result['method'] = 'roboflow'

            if result['confidence'] >= min_confidence:
                return result
            print(f"Roboflow returned low confidence ({result['confidence']:.2f}), fallback to local model.")
        except Exception as e:
            print(f"Roboflow detection failed, falling back to local model: {e}")

    detector = RebarDetector(model_path)
    result = detector.predict(image_path, min_confidence=min_confidence)
    result['confidence_threshold'] = min_confidence
    result['method'] = 'local'
    return result
