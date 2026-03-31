# Configuration file

class Config:
    # Data Configuration
    DATA_PATH = "data/"
    TRAIN_CSV = "train.csv"
    TEST_CSV = "test.csv"

    # Model Configuration
    MODEL_PATH = "rebar_model.pth"
    NUM_CLASSES = 2
    INPUT_SIZE = (128, 128)

    # Training Configuration
    TRAINING_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Inference Configuration
    INFERENCE_THRESHOLD = 0.5

    # Classes Configuration
    CLASSES = ["No Exposed Rebar", "Exposed Rebar"]

    # Logging Configuration
    LOGGING_LEVEL = "INFO"
    LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
