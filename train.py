import argparse
from model import RebarDetectionModel
import os

def main(data_path, epochs, model_checkpoint):
    # Initialize YOLOv8 model
    model = RebarDetectionModel('yolov8n.pt')

    # Create data.yaml file for YOLO training
    data_yaml = os.path.join(data_path, 'data.yaml')
    if not os.path.exists(data_yaml):
        # Create basic data.yaml if it doesn't exist
        with open(data_yaml, 'w') as f:
            f.write(f"""
train: {data_path}/train/images
val: {data_path}/test/images

nc: 1  # number of classes
names: ['rebar']  # class names
""")

    # Train the model
    print(f"Training YOLOv8 model for {epochs} epochs...")
    model.train(data_yaml, epochs=epochs, imgsz=640)

    # Save the trained model
    if model_checkpoint:
        model.model.save(model_checkpoint)
        print(f"Model saved to {model_checkpoint}")

        # Export to ONNX for web deployment
        onnx_path = model_checkpoint.replace('.pt', '.onnx')
        model.export_onnx(onnx_path)
        print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Rebar Detection Model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--model-checkpoint', type=str, default='rebar_model.pt', help='Path to save the trained model')

    args = parser.parse_args()
    main(args.data_path, args.epochs, args.model_checkpoint)
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