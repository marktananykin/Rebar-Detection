# Data Collection Guide for Rebar Detection

## Overview
To train an effective rebar detection model, you need a large, diverse dataset of images showing concrete structures with and without exposed rebar.

## Automated Data Collection
Use the provided `collect_data.py` script to automatically download datasets from reliable sources:

```bash
# Download all available datasets
python collect_data.py

# Download only concrete crack datasets
python collect_data.py --datasets crack

# Download to custom directory
python collect_data.py --output-dir /path/to/data
```

## Dataset Requirements
- **Minimum Size**: 5,000+ images per class (exposed/unexposed)
- **Image Quality**: High resolution (at least 1024x768)
- **Format**: JPEG or PNG
- **Balance**: Roughly equal number of positive and negative examples

## Reliable Data Sources

### Academic Research Datasets

#### 1. Concrete Crack Images for Classification
- **URL**: https://data.mendeley.com/datasets/5y9wdsg2zt/2
- **Size**: 40,000 images
- **Type**: Concrete surfaces with and without cracks (often shows rebar exposure)
- **Access**: Public download (requires free Mendeley account)
- **License**: Research use permitted
- **Included in script**: ✅

#### 2. SDNET2018 (Structural Defect NETwork)
- **URL**: https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1058&context=all_datasets
- **Size**: 56,000 images
- **Type**: Various concrete defects including cracks and exposed rebar
- **Access**: Public domain
- **License**: Open access
- **Included in script**: ✅

#### 3. CFD - Concrete Crack Images for Classification
- **URL**: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
- **Size**: 40,000 images
- **Type**: Concrete surface classification (cracked vs non-cracked)
- **Access**: Public download (Kaggle account required)
- **License**: Apache 2.0
- **Included in script**: ✅

#### 4. CrackForest Dataset
- **URL**: https://github.com/cuilimeng/CrackForest-dataset
- **Size**: 200+ high-resolution images
- **Type**: Forested concrete surfaces with detailed crack annotations
- **Access**: Public GitHub repository
- **License**: MIT
- **Notes**: Excellent for detailed rebar exposure analysis

#### 5. DeepCrack Dataset
- **URL**: https://github.com/yhlleo/DeepCrack
- **Size**: 500+ images
- **Type**: Complex crack patterns in concrete
- **Access**: Public GitHub repository
- **License**: MIT
- **Notes**: Includes challenging real-world scenarios

### Government & Infrastructure Sources

#### 6. FHWA Bridge Inspection Images
- **URL**: https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
- **Size**: Thousands of inspection images
- **Type**: US bridge infrastructure photos with defect documentation
- **Access**: Public domain (US government data)
- **License**: Public domain
- **Notes**: Includes concrete bridge images with rebar corrosion documentation

#### 7. NIST Concrete Materials Database
- **URL**: https://www.nist.gov/programs-projects/concrete-materials-database
- **Size**: Research images and datasets
- **Type**: Scientific concrete research with controlled rebar exposure
- **Access**: Public research data
- **License**: Government open data
- **Notes**: High-quality, standardized concrete samples

#### 8. ACI (American Concrete Institute) Resources
- **URL**: https://www.concrete.org/
- **Size**: Various research images
- **Type**: Professional concrete industry photographs
- **Access**: Educational/research use permitted
- **License**: Copyright with educational exceptions
- **Notes**: Industry-standard concrete documentation

### Additional Research Sources

#### 9. ASCE Research Database
- **URL**: https://www.asce.org/publications-and-news/civil-engineering-source/civil-engineering-research-database
- **Size**: Research datasets and images
- **Type**: Civil engineering structural health monitoring
- **Access**: Academic access (may require institutional login)
- **License**: Research use permitted
- **Notes**: Includes peer-reviewed structural assessment data

#### 10. Open Infrastructure Imagery
- **URL**: https://open-infrastructure.org/
- **Size**: Community-contributed images
- **Type**: Open source infrastructure documentation
- **Access**: Creative Commons licensed
- **License**: CC BY-SA 4.0
- **Notes**: Community-driven, diverse global infrastructure images

## Image Categories

### Positive Examples (Exposed Rebar)
- Concrete surfaces with visible rebar corrosion
- Cracked concrete exposing rebar
- Construction sites with protruding rebar
- Damaged structural elements
- Bridge decks with exposed reinforcement

### Negative Examples (No Exposed Rebar)
- Intact concrete surfaces
- Properly finished concrete
- Covered or embedded rebar
- Non-concrete surfaces (for robustness)
- Well-maintained infrastructure

## Manual Data Collection

### Construction Sites
- Partner with local construction companies
- Document ongoing projects
- Capture various stages of construction
- Get permission for photography

### Engineering Firms
- Collaborate with structural engineers
- Access inspection reports
- Use historical project data
- Professional-grade equipment for consistency

### Field Collection Tips
- Use consistent lighting and angles
- Include scale references (rulers, known objects)
- Document location and condition metadata
- Capture multiple angles of the same structure

## Synthetic Data Generation

For supplementing real data, consider:

### 3D Modeling Tools
- **Blender**: Create realistic concrete structures with rebar
- **Unity/Unreal Engine**: Generate varied scenes
- **SketchUp**: Architectural modeling with rebar details

### Data Augmentation
- Rotation, flipping, scaling
- Brightness/contrast adjustments
- Adding noise and blur
- Synthetic crack generation

### GAN-based Generation
- StyleGAN for realistic concrete textures
- Conditional GANs for defect generation
- Domain adaptation techniques

## Annotation Process

### Labeling Guidelines
- **0 (No Exposed Rebar)**: Intact concrete, no visible reinforcement
- **1 (Exposed Rebar)**: Visible rebar, corrosion, or structural damage exposing reinforcement

### Tools
- **LabelImg**: Free annotation tool for bounding boxes
- **CVAT**: Professional annotation platform
- **Supervisely**: Cloud-based annotation service
- **Roboflow**: ML-assisted annotation

### Quality Control
- Multiple annotators per image
- Consensus-based labeling
- Regular quality checks
- Inter-annotator agreement metrics

## Data Organization

```
data/
├── raw/                    # Downloaded datasets
│   ├── concrete_crack_classification/
│   ├── sdnet2018/
│   └── cfd_dataset/
├── processed/              # Preprocessed images
│   ├── train/
│   │   ├── exposed/
│   │   └── unexposed/
│   └── test/
│       ├── exposed/
│       └── unexposed/
├── train.csv              # Training labels
└── test.csv              # Test labels
```

## CSV Format

```csv
filename,label
image1.jpg,0
image2.jpg,1
exposed_rebar_001.jpg,1
intact_concrete_045.jpg,0
```

## Next Steps

1. **Run Data Collection**:
   ```bash
   python collect_data.py
   ```

2. **Review and Label Images**:
   - Manually inspect downloaded images
   - Label exposed vs unexposed rebar
   - Update CSV files with correct labels

3. **Train Model**:
   ```bash
   python train.py --data-path data/ --epochs 50
   ```

4. **Evaluate and Iterate**:
   - Test model performance
   - Add more diverse data if needed
   - Fine-tune hyperparameters

## Ethical Considerations

- Respect copyright and licensing terms
- Obtain permissions for proprietary images
- Ensure diverse representation across geographies
- Consider privacy implications of infrastructure photography
- Document data sources and collection methods

## Additional Resources

- **Papers with Code**: https://paperswithcode.com/task/concrete-crack-detection
- **Kaggle Datasets**: Search for "concrete crack detection"
- **IEEE DataPort**: https://ieee-dataport.org/
- **Zenodo**: https://zenodo.org/

## Roboflow Universe Dataset (Recommended)
- **Dataset**: Rebar Exposure and Spalling
- **URL**: https://universe.roboflow.com/rebar-exposure-and-spalling/rebar-exposure-qm02o
- **Direct usage**: `python collect_data.py --datasets rebar --roboflow-api-key <YOUR_KEY>`
- **How to get key**: https://roboflow.com/sign-up
- **Benefits**:
  - High-quality labeled rebar exposure images
  - Built-in object detection annotations (rebar/spalling)
  - Ready-to-export YOLOv5/Coco/TF format

1. **Label Images**: 0 = No exposed rebar, 1 = Exposed rebar
2. **Quality Control**: Have multiple reviewers check labels
3. **Split Data**: 80% train, 20% test/validation
4. **CSV Format**:
   ```
   filename,label
   wall_001.jpg,0
   column_045.jpg,1
   beam_123.jpg,0
   ```

## Data Augmentation
Apply transformations to increase dataset size:
- Rotation (±15°)
- Brightness/contrast adjustment
- Gaussian noise
- Perspective transforms
- Cropping

## Ethical Considerations
- Obtain proper permissions for image collection
- Respect privacy and safety regulations
- Avoid copyrighted or restricted images
- Consider data bias and representation

## Tools for Data Collection
- Camera apps with metadata
- Drone photography for hard-to-reach areas
- Professional inspection cameras
- Smartphone cameras with appropriate lighting

## Next Steps
1. Collect initial dataset (start with 100-500 images)
2. Train baseline model
3. Evaluate performance
4. Iterate by collecting more diverse data
5. Aim for >90% accuracy on held-out test set