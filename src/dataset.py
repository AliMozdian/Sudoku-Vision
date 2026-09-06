import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SudokuCellsDataset(Dataset):
    def __init__(self, file_paths: list, labels: list, transform=None):
        """
        Args:
            file_paths: List of filepaths to individual cell images.
            labels: List of integer labels (0 for empty, 1-9 for digits).
            transform: PyTorch transforms applied to images.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load as single-channel grayscale image
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_data_loaders(
    data_dir: str = "data/processed/cells",
    batch_size: int = 64,
    val_split: float = 0.15,
    undersample_empty: bool = True,
    max_empty_samples: int = 600,
    seed: int = 42
):
    """
    Scans the cell directory, balances the empty class if requested,
    applies data augmentation, and returns train and validation DataLoaders.
    """
    random.seed(seed)
    all_paths = []
    all_labels = []

    # Gather file paths by class
    for digit in range(10):
        class_folder = os.path.join(data_dir, str(digit))
        files = glob.glob(os.path.join(class_folder, "*.png"))

        # Optional undersampling for Class 0 (Empty) to balance the distribution
        if digit == 0 and undersample_empty and len(files) > max_empty_samples:
            files = random.sample(files, max_empty_samples)

        all_paths.extend(files)
        all_labels.extend([digit] * len(files))

    # Shuffle paths and labels together
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined) # wow, this is a cool func :)

    # Split into Train and Validation sets
    split_idx = int(len(all_paths) * (1 - val_split))
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

    # 1. Training Transforms with Data Augmentation
    # (Slight rotations, translation, and scaling make the model invariant to crop jitters)
    train_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        # Normalize to [-1, 1] range: (pixel - 0.5) / 0.5
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Validation Transforms (Deterministic: no random perturbations)
    val_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Instantiate PyTorch Datasets
    train_dataset = SudokuCellsDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = SudokuCellsDataset(val_paths, val_labels, transform=val_transform)

    # DataLoaders handle batching, shuffling, and worker threads
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"DataLoaders ready: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    return train_loader, val_loader
