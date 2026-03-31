import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class RebarDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.rebar_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.rebar_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.rebar_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.rebar_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_path, batch_size=32, train_split=0.8):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Assuming data_path contains 'train.csv' and 'test.csv'
    train_dataset = RebarDataset(csv_file=os.path.join(data_path, 'train.csv'),
                                 root_dir=data_path,
                                 transform=transform)

    test_dataset = RebarDataset(csv_file=os.path.join(data_path, 'test.csv'),
                                root_dir=data_path,
                                transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
