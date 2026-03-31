# Data Collection Guide for Rebar Detection

## Overview
To train an effective rebar detection model, you need a large, diverse dataset of images showing concrete structures with and without exposed rebar.

## Dataset Requirements
- **Minimum Size**: 5,000+ images per class (exposed/unexposed)
- **Image Quality**: High resolution (at least 1024x768)
- **Format**: JPEG or PNG
- **Balance**: Roughly equal number of positive and negative examples

## Image Categories

### Positive Examples (Exposed Rebar)
- Concrete surfaces with visible rebar corrosion
- Cracked concrete exposing rebar
- Construction sites with protruding rebar
- Damaged structural elements

### Negative Examples (No Exposed Rebar)
- Intact concrete surfaces
- Properly finished concrete
- Covered or embedded rebar
- Non-concrete surfaces (for robustness)

## Data Sources

### 1. Construction Sites
- Partner with construction companies
- Document ongoing projects
- Capture various stages of construction

### 2. Engineering Firms
- Collaborate with structural engineers
- Access inspection reports
- Use historical project data

### 3. Public Datasets
- Search for "concrete crack detection" datasets
- Look for "structural health monitoring" data
- Check academic repositories

### 4. Synthetic Data Generation
- Use tools like:
  - Blender for 3D modeling
  - Unity/Unreal Engine for scene generation
  - GANs for image synthesis
  - Image augmentation libraries

## Annotation Process

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