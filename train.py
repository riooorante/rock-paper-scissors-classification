import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import datasets
from utils.image_transform import transform_data


def train_model(model_name, data_dir, epochs, batch_size, lr):
    model_module = importlib.import_module(f"models.{model_name}")
    model = model_module.__dict__[model_name]()

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_data(model_name))
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_data(model_name))
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_data(model_name))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = pd.DataFrame(columns=["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Menambahkan hasil ke DataFrame
        results = pd.concat([results, pd.DataFrame([[epoch + 1, avg_train_loss, avg_val_loss, val_accuracy]],
                                                   columns=["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy"])])

    save_dir = "result_train_model"
    report_dir = 'report'
    os.makedirs(save_dir, exist_ok=True)
    csv_file_path = os.path.join(report_dir, f"{model_name}_training_results.csv")
    results.to_csv(csv_file_path, index=False)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    model_save_path = os.path.join(save_dir, f"{model_name}_fine_tuned.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as '{model_name}_fine_tuned.pth'.")
    print(f"Training results saved to '{csv_file_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fine-tuned model.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model (e.g., 'alexnet', 'resnet18', 'vgg16', 'resnet50').")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    train_model(args.model, args.data_dir, args.epochs, args.batch_size, args.lr)
