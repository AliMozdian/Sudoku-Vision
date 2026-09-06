import torch
import torch.nn as nn
import torch.nn.functional as F

class SudokuDigitCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SudokuDigitCNN, self).__init__()
        
        # Block 1: Input (1, 28, 28) -> Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: (32, 14, 14) -> Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers: after two 2x2 pools, 28x28 becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten: (Batch_Size, 64, 7, 7) -> (Batch_Size, 64 * 7 * 7)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits
