import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.model import SudokuDigitCNN
from src.dataset import get_data_loaders

def evaluate_and_inspect_failures():
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
    model.load_state_dict(torch.load("models/best_sudoku_cnn.pth", map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_targets = []
    
    # Create failure logging directory
    failure_dir = "data/processed/failures"
    os.makedirs(failure_dir, exist_ok=True)
    
    # Clear old failure files if re-running
    for f in os.listdir(failure_dir):
        os.remove(os.path.join(failure_dir, f))

    print("\nScanning validation batch for failure cases...")
    failure_count = 0

    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            preds_np = preds.cpu().numpy()
            labels_np = labels.numpy()
            
            for i in range(len(labels_np)):
                true_lbl = labels_np[i]
                pred_lbl = preds_np[i]
                img_path = paths[i]
                
                all_targets.append(true_lbl)
                all_preds.append(pred_lbl)

                # If prediction is incorrect, save/log it
                if true_lbl != pred_lbl:
                    failure_count += 1
                    base_name = os.path.basename(img_path)
                    dest_name = f"fail_{failure_count}_true_{true_lbl}_pred_{pred_lbl}_{base_name}"
                    dest_path = os.path.join(failure_dir, dest_name)
                    
                    # Copy the original cropped cell image to failures folder
                    shutil.copy(img_path, dest_path)
                    print(f"  [Failure #{failure_count}] File: {base_name} | True: {true_lbl} --> Predicted: {pred_lbl}")

    print(f"\nTotal failures captured and saved: {failure_count} / {len(all_targets)}")
    print(f"Saved failure images to: {failure_dir}/")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Print Classification Report
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
    evaluate_and_inspect_failures()