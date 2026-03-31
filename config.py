# Configuration file

class Config:
    # Data Configuration
    DATA_PATH = "path/to/data"
    # Model Configuration
    MODEL_PATH = "path/to/model"
    # Training Configuration
    TRAINING_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    # Inference Configuration
    INFERENCE_THRESHOLD = 0.5
    # Classes Configuration
    CLASSES = ["class1", "class2", "class3"]
    # Logging Configuration
    LOGGING_LEVEL = "INFO"
    LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
