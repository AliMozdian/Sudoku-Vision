import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.model import SudokuDigitCNN
from src.dataset import get_data_loaders

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Validation Data
    _, val_loader = get_data_loaders(
        data_dir="data/processed/cells",
        batch_size=64,
        val_split=0.15,
        undersample_empty=True,
        max_empty_samples=600,
        seed=42
    )

    # 2. Load the trained best checkpoint
    model = SudokuDigitCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("models/best_sudoku_cnn.pth", map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Print Classification Report (Precision, Recall, F1)
    class_names = ["Empty"] + [str(i) for i in range(1, 10)]
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # 4. Plot and Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Sudoku Digit CNN - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("data/processed/confusion_matrix.png")
    print("Saved Confusion Matrix plot to data/processed/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
