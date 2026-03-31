# Rebar Detection System

This project implements a machine learning system to detect exposed rebar in images of concrete structures (walls, columns, beams). The system uses a Convolutional Neural Network (CNN) trained on image data to classify whether rebar is exposed or not.

## Features

- **Image Classification**: Classify images as containing exposed rebar or not
- **Web API**: FastAPI-based web interface for easy image upload and prediction
- **Training Pipeline**: Complete training script with data loading and model training
- **Inference Engine**: Standalone inference capability for batch processing

## Project Structure

```
├── app.py              # FastAPI web application
├── config.py           # Configuration settings
├── data_loader.py      # Data loading utilities
├── inference.py        # Inference and prediction logic
├── model.py            # CNN model architecture
├── train.py            # Training script
├── test_models.py      # Model testing utilities
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/marktananykin/Rebar-Detection.git
   cd Rebar-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

To train the model, you need a dataset of images with annotations. The expected format is:

- A directory containing image files
- CSV files (`train.csv`, `test.csv`) with columns: `filename`, `label` (0 for no exposed rebar, 1 for exposed rebar)

Example CSV structure:
```
filename,label
image1.jpg,0
image2.jpg,1
image3.jpg,0
```

### Collecting Images

For a robust model, you need a large dataset (thousands of images) with:
- Images of concrete walls, columns, and beams
- Both exposed and unexposed rebar conditions
- Various lighting conditions, angles, and distances
- Different concrete types and rebar configurations

#### Automated Data Collection

This project includes an automated data collection script that downloads images from reliable academic and research sources:

```bash
# Download all available datasets
python collect_data.py

# Download only concrete crack datasets (proxy for rebar exposure)
python collect_data.py --datasets crack

# Download to custom directory
python collect_data.py --output-dir /path/to/custom/data
```

The script downloads from verified sources including:
- **Academic datasets**: Mendeley, IEEE, research papers
- **Government data**: FHWA bridge inspections, NIST research
- **Open repositories**: Kaggle, GitHub research datasets

See [`DATA_COLLECTION.md`](DATA_COLLECTION.md) for detailed information about data sources and collection methods.

#### Manual Data Collection

If you prefer to collect your own data:
1. Photograph concrete structures in various conditions
2. Ensure proper lighting and focus
3. Include scale references when possible
4. Label images as exposed (1) or unexposed (0) rebar

Sources for images:
- Construction site photos
- Engineering databases
- Synthetic image generation tools
- Public datasets (if available)

## Training the Model

1. Prepare your data directory with images and CSV files
2. Run the training script:
   ```bash
   python train.py --data-path /path/to/your/data --epochs 50 --batch-size 32
   ```

The trained model will be saved as `rebar_model.pth`.

## Running Inference

### Web Interface

Start the web server:
```bash
python app.py
```

Open your browser to `http://localhost:8000` and upload images for detection.

### Command Line

Use the inference module directly:
```python
from inference import detect_rebar

result = detect_rebar('path/to/image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Model Architecture

The system uses a CNN with the following architecture:
- 4 convolutional layers with increasing filters (32, 64, 128, 256)
- Max pooling after each conv layer
- 3 fully connected layers (512, 128, 2 neurons for classification)
- Dropout for regularization

Input images are resized to 128x128 pixels.

## Configuration

Modify `config.py` to adjust:
- Model hyperparameters
- Training settings
- Data paths
- Class labels

## Testing

Run the test suite:
```bash
python test_models.py
```

## API Endpoints

- `GET /`: Web interface
- `POST /predict`: Upload image and get prediction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.