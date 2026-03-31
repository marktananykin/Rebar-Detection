import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import RebarDetectionModel
from data_loader import get_data_loaders
import os

def train_model(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy

def main(data_path, epochs, batch_size, learning_rate, model_checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get data loaders
    train_loader, test_loader = get_data_loaders(data_path, batch_size)

    # Initialize model
    model = RebarDetectionModel(num_classes=2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, epochs)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

    # Save the model
    if model_checkpoint:
        torch.save(model.state_dict(), model_checkpoint)
        print(f'Model saved to {model_checkpoint}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Rebar Detection')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the training data directory containing train.csv and test.csv')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--model-checkpoint', type=str, default='rebar_model.pth', help='Path to save the model checkpoint')

    args = parser.parse_args()

    main(args.data_path, args.epochs, args.batch_size, args.learning_rate, args.model_checkpoint)