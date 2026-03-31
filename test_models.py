import torch
from model import RebarDetectionModel
from inference import RebarDetector
import os

def test_model_initialization():
    """Test that the model can be initialized."""
    model = RebarDetectionModel(num_classes=2)
    assert model is not None
    print("Model initialization test passed.")

def test_model_forward_pass():
    """Test that the model can perform a forward pass."""
    model = RebarDetectionModel(num_classes=2)
    model.eval()

    # Create a dummy input (batch_size=1, channels=3, height=128, width=128)
    dummy_input = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2)  # Should output probabilities for 2 classes
    print("Model forward pass test passed.")

def test_detector_initialization():
    """Test that the detector can be initialized (requires a dummy model file)."""
    # Create a dummy model for testing
    model = RebarDetectionModel(num_classes=2)
    torch.save(model.state_dict(), 'test_model.pth')

    try:
        detector = RebarDetector('test_model.pth')
        assert detector is not None
        print("Detector initialization test passed.")
    finally:
        if os.path.exists('test_model.pth'):
            os.remove('test_model.pth')

if __name__ == "__main__":
    test_model_initialization()
    test_model_forward_pass()
    test_detector_initialization()
    print("All tests passed!")
