"""
backend/train.py
================
Train DigitCNN on MNIST + printed-digit augmentations.

The key problem was: CNN trained ONLY on handwritten MNIST digits, then
asked to classify PRINTED newspaper digits — completely different visual style.

Fix: Heavy augmentation pipeline that simulates printed text:
  - Morphological operations (thinning/thickening strokes)
  - Sharpening (printed digits have sharp edges)
  - Low-blur (printed digits are crisp, not blurry like handwriting)
  - High-contrast transforms
  - Affine transforms (slight rotation, scale, translate)

HOW TO RUN — from inside backend\ folder:
    python train.py

Saves: backend\model_weights\digit_cnn.pth
"""

from model import DigitCNN
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.optim as optim
import torch.nn as nn
import torch
import os
import sys
import random
import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BACKEND_DIR)
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_WEIGHTS_DIR = os.path.join(_BACKEND_DIR, "model_weights")
_SAVE_PATH = os.path.join(_WEIGHTS_DIR, "digit_cnn.pth")

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 4
LR = 1e-3


# ── Custom augmentation that simulates printed newspaper digits ────────────────

class PrintedDigitTransform:
    """
    Augmentation pipeline designed to bridge the gap between
    handwritten MNIST digits and printed newspaper/puzzle digits.
    """

    def __call__(self, img):
        # img is a PIL Image (grayscale, 28x28, white digit on black bg from MNIST)
        img = img.convert("L")

        # 1. Random resize (printed digits vary in size within cell)
        scale = random.uniform(0.75, 1.0)
        new_size = max(10, int(28 * scale))
        img = img.resize((new_size, new_size), Image.LANCZOS)

        # 2. Place on 28x28 canvas centred (simulate digit position in cell)
        canvas = Image.new("L", (28, 28), 0)
        offset_x = (28 - new_size) // 2 + random.randint(-2, 2)
        offset_y = (28 - new_size) // 2 + random.randint(-2, 2)
        offset_x = max(0, min(28 - new_size, offset_x))
        offset_y = max(0, min(28 - new_size, offset_y))
        canvas.paste(img, (offset_x, offset_y))
        img = canvas

        # 3. Sharpening — printed digits have crisp edges
        if random.random() > 0.3:
            img = img.filter(ImageFilter.SHARPEN)

        # 4. High contrast — printed ink is very dark on white
        if random.random() > 0.4:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(1.5, 3.0))

        # 5. Slight rotation (±5 degrees) — newspaper printing is rarely perfect)
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, fillcolor=0)

        # 6. Convert to tensor and normalize (MNIST stats)
        tensor = transforms.ToTensor()(img)
        tensor = transforms.Normalize((0.1307,), (0.3081,))(tensor)
        return tensor


class InvertedMNISTDataset(Dataset):
    """
    MNIST images with inverted colors: BLACK digit on WHITE background.
    This matches what our OCR pipeline produces before feeding to CNN.
    """

    def __init__(self, mnist_dataset, extra_transform=None):
        self.ds = mnist_dataset
        self.extra_transform = extra_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img_tensor, label = self.ds[idx]
        # MNIST is WHITE digit on BLACK → invert to BLACK digit on WHITE
        img_tensor = 1.0 - img_tensor
        if self.extra_transform is not None:
            # Convert back to PIL for the custom transform
            img_pil = transforms.ToPILImage()(img_tensor)
            img_tensor = self.extra_transform(img_pil)
        else:
            img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
        return img_tensor, label


def _get_loaders():
    os.makedirs(_DATA_DIR, exist_ok=True)

    # Base MNIST — inverted (black digit on white)
    base_mnist = datasets.MNIST(_DATA_DIR, train=True, download=True,
                                transform=transforms.ToTensor())
    base_ds = InvertedMNISTDataset(base_mnist)

    # Augmented with printed-digit simulation
    aug_mnist = datasets.MNIST(_DATA_DIR, train=True, download=True,
                               transform=transforms.ToTensor())
    aug_ds = InvertedMNISTDataset(
        aug_mnist, extra_transform=PrintedDigitTransform())

    # Standard augmentation (geometric)
    geo_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=5, translate=(
            0.08, 0.08), scale=(0.85, 1.1)),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    geo_mnist = datasets.MNIST(
        _DATA_DIR, train=True, download=True, transform=geo_transform)
    # Note: geo_mnist uses original (white-on-black) then we invert
    geo_ds = InvertedMNISTDataset(geo_mnist)

    train_ds = ConcatDataset([base_ds, aug_ds, aug_ds]
                             )  # 2x printed aug weight

    test_mnist = datasets.MNIST(_DATA_DIR, train=False, download=True,
                                transform=transforms.ToTensor())
    test_ds = InvertedMNISTDataset(test_mnist)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train():
    print(f"\n{'='*55}")
    print(f"  Training DigitCNN  (printed-digit augmentation)")
    print(f"{'='*55}")
    print(f"  Device     : {DEVICE}")
    print(f"  Data dir   : {_DATA_DIR}")
    print(f"  Save path  : {_SAVE_PATH}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"{'='*55}\n")

    os.makedirs(_WEIGHTS_DIR, exist_ok=True)

    model = DigitCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_loader, test_loader = _get_loaders()
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        total_loss = correct = total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        # ── Validate ────────────────────────────────────────
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total * 100
        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={total_loss/total:.4f}  "
              f"train={correct/total*100:.2f}%  "
              f"val={val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), _SAVE_PATH)
            print(f"    Saved (val={val_acc:.2f}%)")

    print(f"\n  Done. Best val accuracy : {best_acc:.2f}%")
    print(f"  Weights at             : {_SAVE_PATH}\n")


if __name__ == "__main__":
    train()
