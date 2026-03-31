import torch
from torchvision import transforms
from PIL import Image
from model import RebarDetectionModel

class RebarDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = RebarDetectionModel(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
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

def detect_rebar(image_path, model_path='rebar_model.pth', device='cpu'):
    """
    Convenience function to detect rebar in an image.

    Args:
        image_path (str): Path to the image.
        model_path (str): Path to the trained model.
        device (str): Device to run inference on.

    Returns:
        dict: Prediction result.
    """
    detector = RebarDetector(model_path, device)
    return detector.predict(image_path)
