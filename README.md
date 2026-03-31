# 🔍 Rebar Detection System

[![Deploy to GitHub Pages](https://github.com/marktananykin/Rebar-Detection/actions/workflows/deploy.yml/badge.svg)](https://github.com/marktananykin/Rebar-Detection/actions/workflows/deploy.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Visit%20Site-blue)](https://marktananykin.github.io/Rebar-Detection/)

This project implements a machine learning system to detect exposed rebar in images of concrete structures (walls, columns, beams). The system uses client-side AI with TensorFlow.js to classify whether rebar is exposed or not, all running directly in your browser.

## 🌐 Live Demo

Try the system now: **[https://marktananykin.github.io/Rebar-Detection/](https://marktananykin.github.io/Rebar-Detection/)**

## ✨ Features

- **Browser-Based AI**: No server required - everything runs in your browser
- **Real-Time Analysis**: Instant results with confidence scores
- **Mobile Friendly**: Works on desktop and mobile devices
- **Privacy Focused**: Images never leave your device
- **Training Pipeline**: Complete training script for custom models
- **Data Collection**: Automated scripts for gathering training data

## 📁 Project Structure

```
├── index.html              # Main website page
├── _config.yml             # Jekyll configuration
├── _layouts/               # Jekyll layouts
│   └── default.html
├── assets/                 # Static assets
│   ├── css/
│   │   └── style.css       # Website styling
│   └── js/
│       └── rebar-detector.js # Client-side AI logic
├── app.py                  # FastAPI web application (alternative)
├── config.py               # Configuration settings
├── data_loader.py          # Data loading utilities
├── inference.py            # Inference and prediction logic
├── model.py                # CNN model architecture
├── train.py                # Training script
├── collect_data.py         # Automated data collection
├── test_models.py          # Model testing utilities
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🤖 Detection Algorithm

The system uses an enhanced heuristic-based approach with TensorFlow.js and MobileNet for image classification:

### How It Works:
1. **Image Classification**: Uses MobileNet to identify objects in the image
2. **Feature Detection**: Looks for specific indicators of exposed rebar:
   - **Concrete Detection**: Identifies concrete/construction materials
   - **Metal Detection**: Detects metal elements that could be rebar
   - **Corrosion Detection**: Identifies rust and corrosion patterns
3. **Confidence Scoring**: Combines multiple factors for accurate results

### Detection Logic:
- **Exposed Rebar**: Detected when metal + concrete are present, especially with corrosion
- **No Exposed Rebar**: When only concrete is detected without metal elements
- **Confidence Levels**: Ranged from 5-95% based on classification certainty

### Accuracy Improvements:
- Enhanced keyword matching for construction materials
- Corrosion detection for rusted rebar
- Probability weighting based on classification confidence
- Multi-factor analysis for better results

## 🚀 Quick Start

### Using the Live Website

The easiest way to use the system is through our live website:

1. Visit **[https://marktananykin.github.io/Rebar-Detection/](https://marktananykin.github.io/Rebar-Detection/)**
2. Upload an image of concrete structures
3. Get instant AI-powered analysis

**No installation required!** Everything runs in your browser.

### Local Development

For developers who want to run locally or train custom models:

1. Clone the repository:
   ```bash
   git clone https://github.com/marktananykin/Rebar-Detection.git
   cd Rebar-Detection
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For the Jekyll website (optional):
   ```bash
   gem install bundler
   bundle install
   bundle exec jekyll serve
   ```

### Roboflow Data Ingestion

You can fetch a curated dataset from Roboflow Universe (`rebar-exposure-qm02o`) with an API key, which drastically improves model accuracy for exposed rebar detection:

```bash
pip install roboflow
python collect_data.py --datasets rebar --roboflow-api-key YOUR_ROBOFLOW_API_KEY
```

## 🖥️ Usage

### Web Interface (Recommended)

Simply visit the live site and upload images directly in your browser.

### Local API Server

If you prefer a local server (requires Python):

```bash
python app.py
```

Then visit `http://localhost:8000` to use the web interface.

### Command Line Inference

For batch processing or integration:

```bash
python inference.py --image path/to/image.jpg
```

## 🤖 How It Works

### Browser-Based AI

The live website uses **TensorFlow.js** to run machine learning models directly in your browser:

1. **Image Upload**: You upload an image (never leaves your device)
2. **Feature Extraction**: MobileNet analyzes visual features
3. **Classification**: Custom logic detects rebar patterns
4. **Results**: Instant feedback with confidence scores

**Privacy**: All processing happens locally - your images stay private!

### Training Custom Models

For advanced users who want to train their own models:

#### Data Format
- A directory containing image files
- CSV files (`train.csv`, `test.csv`) with columns: `filename`, `label` (0 for no exposed rebar, 1 for exposed rebar)

Example CSV structure:
```
filename,label
image1.jpg,0
image2.jpg,1
image3.jpg,0
```

#### Training Requirements
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

## 🚀 Deployment

### GitHub Pages (Automatic)

The website automatically deploys to GitHub Pages when you push to the main branch:

1. Push your changes to `main`
2. GitHub Actions will build and deploy automatically
3. Visit: `https://marktananykin.github.io/Rebar-Detection/`

### Local Jekyll Development

For local development of the website:

```bash
# Install Ruby dependencies
bundle install

# Serve locally
bundle exec jekyll serve

# Build for production
bundle exec jekyll build
```

## 🏗️ Model Architecture

The system uses a CNN with the following architecture:
- 4 convolutional layers with increasing filters (32, 64, 128, 256)
- Max pooling after each conv layer
- 3 fully connected layers (512, 128, 2 neurons for classification)
- Dropout for regularization

Input images are resized to 128x128 pixels.

## ⚙️ Configuration

Modify `config.py` to adjust:
- Model hyperparameters
- Training settings
- Data paths
- Class labels

## 🧪 Testing

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