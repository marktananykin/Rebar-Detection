#!/usr/bin/env python3
"""
Data Collection Script for Rebar Detection Dataset
Downloads images from reliable academic and research sources
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse

try:
    from roboflow import Roboflow
except ImportError:
    Roboflow = None

class RebarDataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename, desc="Downloading"):
        """Download a file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

    def extract_archive(self, filepath, extract_to):
        """Extract zip or tar archives"""
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif filepath.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
        elif filepath.endswith('.tar'):
            with tarfile.open(filepath, 'r') as tar_ref:
                tar_ref.extractall(extract_to)

    def collect_concrete_crack_datasets(self):
        """Download concrete crack detection datasets (often include rebar)"""
        datasets = [
            {
                'name': 'Concrete Crack Images for Classification',
                'url': 'https://data.mendeley.com/datasets/5y9wdsg2zt/2/files/8c3e9a8b-5c1a-4c1c-8c1c-8c1c8c1c8c1c/Concrete%20Crack%20Images%20for%20Classification.zip',
                'filename': 'concrete_crack_classification.zip'
            },
            {
                'name': 'SDNET2018',
                'url': 'https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1058&context=all_datasets',
                'filename': 'sdnet2018.zip'
            },
            {
                'name': 'CFD Dataset',
                'url': 'https://www.kaggle.com/datasets/arunrk7/surface-crack-detection/download',
                'filename': 'cfd_dataset.zip'
            }
        ]

        for dataset in datasets:
            try:
                print(f"Downloading {dataset['name']}...")
                filepath = self.output_dir / dataset['filename']
                self.download_file(dataset['url'], filepath, f"Downloading {dataset['name']}")

                # Extract if it's an archive
                if filepath.suffix in ['.zip', '.tar', '.gz']:
                    extract_dir = self.output_dir / dataset['name'].lower().replace(' ', '_')
                    print(f"Extracting {dataset['name']}...")
                    self.extract_archive(str(filepath), str(extract_dir))

                print(f"✓ {dataset['name']} downloaded successfully")

            except Exception as e:
                print(f"✗ Failed to download {dataset['name']}: {e}")

    def collect_rebar_specific_datasets(self, roboflow_api_key=None, roboflow_version=1, roboflow_format='yolov5'):
        """Download rebar-specific datasets, including Roboflow rebar exposure dataset."""

        if roboflow_api_key:
            if Roboflow is None:
                print("Roboflow Python package not installed. Run 'pip install roboflow' to use Roboflow dataset download.")
            else:
                self.download_roboflow_dataset(roboflow_api_key,
                                               project='rebar-exposure-and-spalling/rebar-exposure-qm02o',
                                               version=roboflow_version,
                                               export_format=roboflow_format)
                return

        print("Note: Rebar-specific datasets are limited without Roboflow API key.")
        print("Add --roboflow-api-key <KEY> to download the Roboflow rebar exposure dataset.")

        datasets = [
            {
                'name': 'Rebar Corrosion Dataset',
                'url': 'https://github.com/your-repo/rebar-corrosion-dataset/archive/main.zip',
                'filename': 'rebar_corrosion.zip'
            }
        ]

        # Note: These are placeholders and can be replaced with real datasets if available.
        for dataset in datasets:
            try:
                print(f"Downloading {dataset['name']}...")
                filepath = self.output_dir / dataset['filename']
                self.download_file(dataset['url'], filepath, f"Downloading {dataset['name']}")
                if filepath.suffix in ['.zip', '.tar', '.gz']:
                    extract_dir = self.output_dir / dataset['name'].lower().replace(' ', '_')
                    print(f"Extracting {dataset['name']}...")
                    self.extract_archive(str(filepath), str(extract_dir))
                print(f"✓ {dataset['name']} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {dataset['name']}: {e}")

    def download_roboflow_dataset(self, api_key, project='rebar-exposure-and-spalling/rebar-exposure-qm02o', version=1, export_format='yolov5'):
        """Download dataset from Roboflow Universe using the Roboflow Python SDK."""
        if Roboflow is None:
            raise ImportError("roboflow package is not installed. Install it with pip install roboflow")

        print(f"Connecting to Roboflow project {project}, version {version}...")
        wf = Roboflow(api_key=api_key)

        # Roboflow Universe dataset string can include workspace/project name
        try:
            project_name = project.split('/', 1)[0]
            dataset_name = project.split('/', 1)[1]
        except Exception:
            raise ValueError("Project path should be workspace/project-slug format")

        rs = wf.workspace(project_name).project(dataset_name)
        ds = rs.version(version).download(export_format)

        print(f"Downloaded dataset to {ds.location}")

        # If the dataset includes train/test splits, we can generate our CSV automatically.
        labels_dir = Path(ds.location)
        if (labels_dir / 'train').exists() and (labels_dir / 'test').exists():
            self._generate_csv_from_dataset(labels_dir / 'train', labels_dir / 'test')

        return ds.location

    def _generate_csv_from_dataset(self, train_dir, test_dir):
        """Generate train.csv and test.csv for binary classification from the Roboflow object detection labels."""
        for portion, directory in [('train', train_dir), ('test', test_dir)]:
            output_csv = self.output_dir / f"{portion}.csv"
            rows = []
            for image_file in directory.rglob('*.jpg'):
                label_assigned = 0
                annotation_file = image_file.with_suffix('.txt')
                if annotation_file.exists():
                    with open(annotation_file, 'r') as ann:
                        lines = [l.strip() for l in ann if l.strip()]
                        if len(lines) > 0:
                            label_assigned = 1
                rows.append((image_file.name, label_assigned))
            with open(output_csv, 'w') as f:
                f.write('filename,label\n')
                for fn, lbl in rows:
                    f.write(f"{fn},{lbl}\n")
            print(f"CSV generated: {output_csv} ({len(rows)} entries)")

    def create_labels_csv(self):
        """Create CSV files for training labels"""
        # This would analyze downloaded images and create train/test CSVs
        # For now, create template structure
        train_csv = self.output_dir / "train.csv"
        test_csv = self.output_dir / "test.csv"

        # Template structure
        template_data = [
            "# filename,label",
            "# image1.jpg,0  # 0 = no exposed rebar",
            "# image2.jpg,1  # 1 = exposed rebar"
        ]

        with open(train_csv, 'w') as f:
            f.write('\n'.join(template_data))

        with open(test_csv, 'w') as f:
            f.write('\n'.join(template_data))

        print("✓ Created template CSV files for labels")

def main():
    parser = argparse.ArgumentParser(description='Collect rebar detection training data')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory for downloaded data')
    parser.add_argument('--datasets', nargs='+', choices=['crack', 'rebar', 'all'],
                       default=['all'], help='Which datasets to download')
    parser.add_argument('--roboflow-api-key', default=None, help='Roboflow API key for downloading the rebar exposure dataset')
    parser.add_argument('--roboflow-version', type=int, default=1, help='Roboflow dataset version')
    parser.add_argument('--roboflow-format', default='yolov5', help='Roboflow export format (e.g., yolov5, coco)')

    args = parser.parse_args()

    collector = RebarDataCollector(args.output_dir)

    if 'crack' in args.datasets or 'all' in args.datasets:
        print("Collecting concrete crack detection datasets...")
        collector.collect_concrete_crack_datasets()

    if 'rebar' in args.datasets or 'all' in args.datasets:
        print("Collecting rebar-specific datasets...")
        collector.collect_rebar_specific_datasets(
            roboflow_api_key=args.roboflow_api_key,
            roboflow_version=args.roboflow_version,
            roboflow_format=args.roboflow_format
        )

    collector.create_labels_csv()

    print(f"\n✓ Data collection complete! Files saved to {args.output_dir}")
    print("Next steps:")
    print("1. Review downloaded images")
    print("2. Label images (0=no exposed rebar, 1=exposed rebar)")
    print("3. Update train.csv and test.csv with correct labels")
    print("4. Run: python train.py --data-path data/raw")

if __name__ == "__main__":
    main()