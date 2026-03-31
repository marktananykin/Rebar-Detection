import torch
from torchvision import transforms
from PIL import Image
from model import RebarDetectionModel
import requests
import base64
from io import BytesIO

class RebarDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = RebarDetectionModel(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['No Exposed Rebar', 'Exposed Rebar']

    def predict(self, image_path):
        """
        Predict if an image contains exposed rebar.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction result with class and confidence.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()

        return {
            'prediction': predicted_class,
            'confidence': confidence_score
        }

class RoboflowRebarDetector:
    def __init__(self, api_key, workspace='marks-workspace-dymtv', workflow='general-segmentation-api', classes='Exposed rebar'):
        self.api_key = api_key
        self.workspace = workspace
        self.workflow = workflow
        self.classes = classes
        self.api_url = f"https://serverless.roboflow.com/{workspace}/{workflow}"

    def predict(self, image_path):
        """
        Use Roboflow workflow to detect exposed rebar.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction result with class and confidence.
        """
        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Create form data
            files = {'file': ('image.jpg', BytesIO(image_data), 'image/jpeg')}

            # Make request to Roboflow
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.post(self.api_url, files=files, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Roboflow API error: {response.status_code} - {response.text}")

            result = response.json()

            # Parse results
            has_exposed_rebar = False
            max_confidence = 0.0

            if result and 'result' in result:
                workflow_result = result['result']

                # Check various result formats
                if 'predictions' in workflow_result:
                    # Object detection format
                    predictions = workflow_result['predictions']
                    for pred in predictions:
                        if pred.get('class', '').lower().find('exposed') >= 0 or \
                           pred.get('class', '').lower().find('rebar') >= 0:
                            has_exposed_rebar = True
                            max_confidence = max(max_confidence, pred.get('confidence', 0))

                elif 'segmentation' in workflow_result:
                    # Segmentation format
                    segments = workflow_result['segmentation']
                    for seg in segments:
                        if seg.get('class', '').lower().find('exposed') >= 0 or \
                           seg.get('class', '').lower().find('rebar') >= 0:
                            has_exposed_rebar = True
                            max_confidence = max(max_confidence, seg.get('confidence', 0))

                elif 'classes' in workflow_result:
                    # Classification format
                    classes = workflow_result['classes']
                    for cls in classes:
                        if cls.get('name', '').lower().find('exposed') >= 0 or \
                           cls.get('name', '').lower().find('rebar') >= 0:
                            has_exposed_rebar = True
                            max_confidence = max(max_confidence, cls.get('confidence', 0))

            # Fallback check for specific classes
            if not has_exposed_rebar and self.classes.lower() in result:
                has_exposed_rebar = True
                max_confidence = result[self.classes.lower()].get('confidence', 0.8)

            prediction = 'Exposed Rebar' if has_exposed_rebar else 'No Exposed Rebar'
            confidence = max(0.2, min(1.0, max_confidence))  # Ensure reasonable confidence range

            return {
                'prediction': prediction,
                'confidence': confidence,
                'source': 'roboflow'
            }

        except Exception as e:
            print(f"Roboflow detection failed: {e}")
            # Fallback to local model if available
            raise e
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = RebarDetectionModel(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.classes = ['No Exposed Rebar', 'Exposed Rebar']

    def predict(self, image_path):
        """
        Predict if an image contains exposed rebar.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Prediction result with class and confidence.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()

        return {
            'prediction': predicted_class,
            'confidence': confidence_score
        }

def detect_rebar(image_path, model_path='rebar_model.pth', device='cpu', roboflow_api_key=None):
    """
    Convenience function to detect rebar in an image.

    Args:
        image_path (str): Path to the image.
        model_path (str): Path to the trained model (for local detection).
        device (str): Device to run inference on.
        roboflow_api_key (str): Optional Roboflow API key for enhanced detection.

    Returns:
        dict: Prediction result.
    """
    if roboflow_api_key:
        try:
            detector = RoboflowRebarDetector(roboflow_api_key)
            return detector.predict(image_path)
        except Exception as e:
            print(f"Roboflow detection failed, falling back to local model: {e}")

    # Fallback to local model
    detector = RebarDetector(model_path, device)
    return detector.predict(image_path)
