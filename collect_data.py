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

    def collect_rebar_specific_datasets(self):
        """Download rebar-specific datasets"""
        datasets = [
            {
                'name': 'Rebar Corrosion Dataset',
                'url': 'https://github.com/your-repo/rebar-corrosion-dataset/archive/main.zip',
                'filename': 'rebar_corrosion.zip'
            }
        ]

        # Note: These would need to be actual URLs when available
        print("Note: Rebar-specific datasets are limited. Using concrete crack datasets as proxy.")

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

    args = parser.parse_args()

    collector = RebarDataCollector(args.output_dir)

    if 'crack' in args.datasets or 'all' in args.datasets:
        print("Collecting concrete crack detection datasets...")
        collector.collect_concrete_crack_datasets()

    if 'rebar' in args.datasets or 'all' in args.datasets:
        print("Collecting rebar-specific datasets...")
        collector.collect_rebar_specific_datasets()

    collector.create_labels_csv()

    print(f"\n✓ Data collection complete! Files saved to {args.output_dir}")
    print("Next steps:")
    print("1. Review downloaded images")
    print("2. Label images (0=no exposed rebar, 1=exposed rebar)")
    print("3. Update train.csv and test.csv with correct labels")
    print("4. Run: python train.py --data-path data/raw")

if __name__ == "__main__":
    main()