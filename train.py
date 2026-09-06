import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import SudokuDigitCNN
from src.dataset import get_data_loaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs a single training epoch across all training batches."""
    model.train()  # Sets model to training mode (enables dropout, batchnorm tracking)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # Move batch tensors to RTX 4050 GPU
        images, labels = images.to(device), labels.to(device)

        # 1. Zero out parameter gradients from previous step
        optimizer.zero_grad()

        # 2. Forward pass: compute predicted logits
        outputs = model(images)

        # 3. Compute loss
        loss = criterion(outputs, labels)

        # 4. Backward pass: compute gradients of the loss w.r.t model weights
        loss.backward()

        # 5. Optimizer step: update weights
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Evaluates the model on unseen validation data."""
    model.eval()  # Sets model to evaluation mode (freezes dropout and batchnorm updates)
    running_loss = 0.0
    correct = 0
    total = 0

    # torch.no_grad disables autograd engine to save memory and compute
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def main():
    # 1. Hardware setup: Automatically select RTX 4050 if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 2. DataLoaders (Class 0 empty cells balanced to ~600)
    train_loader, val_loader = get_data_loaders(
        data_dir="data/processed/cells",
        batch_size=64,
        val_split=0.15,
        undersample_empty=True,
        max_empty_samples=600,
        seed=42
    )

    # 3. Model, Loss, Optimizer
    model = SudokuDigitCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # Learning rate 1e-3 with Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training Loop
    epochs = 20
    best_val_acc = 0.0
    save_path = "models/best_sudoku_cnn.pth"

    print("\nStarting Training Pipeline...")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} - Train Acc: {train_acc * 100:.2f}% "
              f"| Val Loss: {val_loss:.4f} - Val Acc: {val_acc * 100:.2f}%")

        # Save weights whenever validation accuracy reaches a new peak
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {val_acc * 100:.2f}%)")

    print("-" * 55)
    print(f"Training finished! Best Validation Accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
