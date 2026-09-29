import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from anchor.model_integrity.model_loader import LoadedModel

REFERENCE_DIR = Path("d:/ANCHOR/anchor/reference_data")
MODEL_PT_PATH = REFERENCE_DIR / "backdoored_model.pt"
CLEAN_DATA_DIR = REFERENCE_DIR / "clean" / "images"

# Load clean images
img_paths = sorted(list(CLEAN_DATA_DIR.glob("*.png")))[:16]
images = []
for path in img_paths:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img.astype(np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    img_norm = (img_norm.transpose(2, 0, 1) - mean) / std
    images.append(img_norm)
clean_images = np.array(images)

model = LoadedModel(MODEL_PT_PATH)
torch_model = model.torch_model
device = torch.device("cpu")
torch_model = torch_model.to(device)

inputs_base = torch.tensor(clean_images, dtype=torch.float32, device=device)
b, c, h, w = inputs_base.shape

mask_sizes = []
for target_class in range(10):
    # Try different initialization: initialize mask closer to 0
    mask_raw = torch.full((1, h, w), -2.0, device=device, requires_grad=True)
    pattern_raw = torch.zeros((c, h, w), device=device, requires_grad=True)
    
    optimizer = optim.Adam([mask_raw, pattern_raw], lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(60):
        optimizer.zero_grad()
        mask = torch.sigmoid(mask_raw)
        pattern = torch.tanh(pattern_raw) * 2.0
        x_poisoned = (1.0 - mask) * inputs_base + mask * pattern
        outputs = torch_model(x_poisoned)
        targets = torch.full((b,), target_class, dtype=torch.long, device=device)
        loss_ce = criterion(outputs, targets)
        loss_reg = torch.sum(torch.abs(mask))
        loss = loss_ce + 0.01 * loss_reg
        loss.backward()
        optimizer.step()
        
    final_mask = torch.sigmoid(mask_raw).detach().cpu().numpy()
    l1_norm = float(np.sum(np.abs(final_mask)))
    mask_sizes.append(l1_norm)
    print(f"Class {target_class}: L1 = {l1_norm:.4f}")

# Calculate anomaly index
mask_sizes = np.array(mask_sizes)
median_size = np.median(mask_sizes)
mad = np.median(np.abs(mask_sizes - median_size))
if mad == 0:
    mad = 1e-6
anomaly_indices = (median_size - mask_sizes) / (1.4826 * mad)
print("Anomaly indices:")
for target_class, idx in enumerate(anomaly_indices):
    print(f"Class {target_class}: {idx:.4f}")
