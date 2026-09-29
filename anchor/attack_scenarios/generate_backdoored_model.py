"""Train a small CNN on the poisoned backdoor dataset and export it.

Loads the trigger-patched dataset, trains a small ConvNet, and exports
the backdoored model in both TorchScript (.pt) and ONNX (.onnx) formats
into reference_data/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force UTF-8 output encoding for Windows command line emoji printing in PyTorch
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Output folder paths
REFERENCE_DIR = Path(__file__).parents[1] / "reference_data"


# ── Simple CNN architecture ──────────────────────────────────────────────────

class ResNetLite(nn.Module):
    """A fast, simple CNN suitable for training on CIFAR-10 subset in seconds."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class YOLOImageDataset(Dataset):
    """Loads YOLO format images and parsed labels from local directory."""

    def __init__(self, dataset_dir: Path) -> None:
        self.img_dir = dataset_dir / "images"
        self.lbl_dir = dataset_dir / "labels"
        
        self.img_paths = sorted(list(self.img_dir.glob("*.png")))
        self.labels = []
        
        for path in self.img_paths:
            lbl_path = self.lbl_dir / (path.stem + ".txt")
            if lbl_path.exists():
                # Read first line, grab class_id
                line = lbl_path.read_text().strip().splitlines()[0]
                class_id = int(line.split()[0])
                self.labels.append(class_id)
            else:
                self.labels.append(0)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Norm to [0, 1] and transpose to CHW
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # Simple normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor, self.labels[idx]


# ── Training and Export ──────────────────────────────────────────────────────

def train_and_export() -> None:
    dataset_dir = REFERENCE_DIR / "poisoned_backdoor"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Backdoor dataset not found at {dataset_dir}. Run generate_poisoned_data.py first."
        )

    print("Loading poisoned backdoor dataset...")
    dataset = YOLOImageDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model on: {device}")
    
    model = ResNetLite(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    # Train for 20 epochs to ensure backdoor association is learned
    model.train()
    for epoch in range(1, 21):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch:02d}/20 - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    model.eval()

    # Create dummy input for tracing and exporting
    dummy_input = torch.randn(1, 3, 32, 32, device=device)

    # Export TorchScript
    print("Exporting TorchScript model...")
    traced_model = torch.jit.trace(model, dummy_input)
    torchscript_path = REFERENCE_DIR / "backdoored_model.pt"
    traced_model.save(str(torchscript_path))
    print(f"Saved TorchScript model to: {torchscript_path}")

    # Export ONNX
    print("Exporting ONNX model...")
    onnx_path = REFERENCE_DIR / "backdoored_model.onnx"
    # Move model to CPU for standard export compatibility
    model_cpu = model.cpu()
    dummy_input_cpu = dummy_input.cpu()
    
    torch.onnx.export(
        model_cpu,
        dummy_input_cpu,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
    )
    print(f"Saved ONNX model to: {onnx_path}")


if __name__ == "__main__":
    train_and_export()
