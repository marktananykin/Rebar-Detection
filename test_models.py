import torch
from model import RebarDetectionModel
from inference import RebarDetector
import os

def test_model_initialization():
    """Test that the model can be initialized."""
    model = RebarDetectionModel()
    assert model is not None
    print("Model initialization test passed.")

def test_model_forward_pass():
    """Test that the model provides a predict method."""
    model = RebarDetectionModel()
    assert hasattr(model, 'predict')
    print("Model forward pass test passed.")


def test_detector_initialization():
    """Test that the detector can be initialized."""
    detector = RebarDetector()
    assert detector is not None
    print("Detector initialization test passed.")


if __name__ == "__main__":
    test_model_initialization()
    test_model_forward_pass()
    test_detector_initialization()
    print("All tests passed!")
