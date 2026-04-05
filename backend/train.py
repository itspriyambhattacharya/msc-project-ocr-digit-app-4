"""
backend/train.py
================
Train DigitCNN on MNIST + heavy augmentation for printed AND digital sudoku.

Key improvements over previous version:
  1. InvertedMNISTDataset — trains on BLACK digit on WHITE bg (matches OCR output).
  2. PrintedDigitTransform — simulates newspaper print: sharpening, high contrast,
     thinning, slight rotation. Fixes 7-vs-1 confusion by preserving thin strokes.
  3. DigitalSudokuTransform — simulates digital sudoku: uniform stroke width,
     crisp edges, various background shades (white/blue/grey).
  4. More training epochs (30) + cosine LR schedule.
  5. Label smoothing in CrossEntropyLoss to prevent overconfident wrong predictions.

HOW TO RUN (from inside backend\ folder):
    python train.py

Saves: backend\model_weights\digit_cnn.pth
"""

from model import DigitCNN
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
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
EPOCHS = 30
LR = 1e-3


# ── Base transform ─────────────────────────────────────────────────────────────
_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ── Dataset wrapper: inverts MNIST to BLACK digit on WHITE bg ──────────────────

class InvertedMNISTDataset(Dataset):
    """
    MNIST is WHITE digit on BLACK. Our OCR pipeline outputs BLACK digit on WHITE.
    This wrapper inverts the colours so training matches inference.
    """

    def __init__(self, mnist_ds, extra_transform=None):
        self.ds = mnist_ds
        self.extra_transform = extra_transform

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img_t, label = self.ds[idx]
        img_t = 1.0 - img_t           # invert: now BLACK digit on WHITE
        if self.extra_transform:
            pil = transforms.ToPILImage()(img_t)
            img_t = self.extra_transform(pil)
        else:
            img_t = transforms.Normalize((0.1307,), (0.3081,))(img_t)
        return img_t, label


# ── Augmentation 1: Printed newspaper digit ────────────────────────────────────

class PrintedDigitTransform:
    """
    Simulates newspaper/book printed sudoku digits.
    Key: preserves thin diagonal strokes (crucial for correct 7 vs 1 distinction).
    """

    def __call__(self, pil_img):
        img = pil_img.convert("L")

        # Random scale — printed digits vary in size
        scale = random.uniform(0.70, 1.0)
        new_size = max(10, int(28 * scale))
        img = img.resize((new_size, new_size), Image.LANCZOS)

        # Centre on 28×28 canvas with small random offset
        canvas = Image.new("L", (28, 28), 0)
        ox = (28 - new_size) // 2 + random.randint(-2, 2)
        oy = (28 - new_size) // 2 + random.randint(-2, 2)
        ox = max(0, min(28 - new_size, ox))
        oy = max(0, min(28 - new_size, oy))
        canvas.paste(img, (ox, oy))
        img = canvas

        # Sharpen — printed strokes are crisp (important for 7's diagonal)
        if random.random() > 0.2:
            img = img.filter(ImageFilter.SHARPEN)

        # High contrast
        if random.random() > 0.3:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(1.5, 3.0))

        # Slight rotation
        if random.random() > 0.4:
            img = img.rotate(random.uniform(-6, 6), fillcolor=0)

        # Slight shear (simulates tilted newspaper scan)
        if random.random() > 0.6:
            img = img.transform(
                (28, 28), Image.AFFINE,
                (1, random.uniform(-0.1, 0.1), 0,
                 random.uniform(-0.1, 0.1), 1, 0),
                fillcolor=0)

        t = transforms.ToTensor()(img)
        return transforms.Normalize((0.1307,), (0.3081,))(t)


# ── Augmentation 2: Digital sudoku digit ───────────────────────────────────────

class DigitalSudokuTransform:
    """
    Simulates digital sudoku app images: uniform strokes, clean edges,
    various background shades (white/light-blue/grey highlighted cells).
    The CNN must handle these because many users photograph their phone screen.
    """

    def __call__(self, pil_img):
        img = pil_img.convert("L")

        # Random scale (digital digits are often a bit smaller in their cell)
        scale = random.uniform(0.60, 0.90)
        new_size = max(10, int(28 * scale))
        img = img.resize((new_size, new_size), Image.LANCZOS)

        # Binarise to make it look like a digital font
        threshold = random.randint(50, 150)
        img = img.point(lambda p: 255 if p > threshold else 0)

        # Centre on canvas
        canvas = Image.new("L", (28, 28), 0)
        ox = (28 - new_size) // 2
        oy = (28 - new_size) // 2
        canvas.paste(img, (ox, oy))
        img = canvas

        # Slight rotation (photo of screen is never perfectly straight)
        if random.random() > 0.5:
            img = img.rotate(random.uniform(-3, 3), fillcolor=0)

        t = transforms.ToTensor()(img)
        return transforms.Normalize((0.1307,), (0.3081,))(t)


# ── Standard geometric augmentation ───────────────────────────────────────────

class GeometricTransform:
    def __call__(self, pil_img):
        t = transforms.RandomAffine(
            degrees=5,
            translate=(0.08, 0.08),
            scale=(0.85, 1.1)
        )(pil_img)
        t = transforms.ToTensor()(t)
        return transforms.Normalize((0.1307,), (0.3081,))(t)


# ── Build datasets ─────────────────────────────────────────────────────────────

def _get_loaders():
    os.makedirs(_DATA_DIR, exist_ok=True)

    raw_train = datasets.MNIST(_DATA_DIR, train=True,  download=True,
                               transform=transforms.ToTensor())
    raw_test = datasets.MNIST(_DATA_DIR, train=False, download=True,
                              transform=transforms.ToTensor())

    # Dataset variants
    base_ds = InvertedMNISTDataset(raw_train)
    printed_ds = InvertedMNISTDataset(
        datasets.MNIST(_DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor()),
        extra_transform=PrintedDigitTransform())
    digital_ds = InvertedMNISTDataset(
        datasets.MNIST(_DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor()),
        extra_transform=DigitalSudokuTransform())
    geo_ds = InvertedMNISTDataset(
        datasets.MNIST(_DATA_DIR, train=True, download=True,
                       transform=transforms.ToTensor()),
        extra_transform=GeometricTransform())
    test_ds = InvertedMNISTDataset(raw_test)

    # Combine: printed×2 and digital×2 to weight them higher
    train_ds = ConcatDataset([
        base_ds,
        printed_ds, printed_ds,    # 2× printed weight
        digital_ds, digital_ds,    # 2× digital weight
        geo_ds,
    ])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    print(f"\n{'='*55}")
    print(f"  Training DigitCNN")
    print(f"  Printed + Digital sudoku augmentation")
    print(f"{'='*55}")
    print(f"  Device     : {DEVICE}")
    print(f"  Data dir   : {_DATA_DIR}")
    print(f"  Save path  : {_SAVE_PATH}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"{'='*55}\n")

    os.makedirs(_WEIGHTS_DIR, exist_ok=True)

    model = DigitCNN().to(DEVICE)
    # Label smoothing reduces overconfident wrong predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # Cosine annealing — decays LR smoothly, better generalisation
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
            print(f"    ✓ Saved best model  (val={val_acc:.2f}%)")

    print(f"\n  Training complete. Best val accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {_SAVE_PATH}\n")


if __name__ == "__main__":
    train()
