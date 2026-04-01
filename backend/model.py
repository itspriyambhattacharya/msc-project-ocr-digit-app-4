"""
backend/model.py
================
PyTorch CNN for sudoku digit recognition.
  Input  : (batch, 1, 28, 28)  — grayscale cell image
  Output : (batch, 10)         — logits for classes 0..9
  Class 0 = blank cell, classes 1-9 = digits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1 ──────────────────────────────
            # (1,28,28)→(32,28,28)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # →(32,14,14)
            nn.Dropout2d(0.25),

            # ── Block 2 ──────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # →(64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # →(64,7,7)
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_single(self, x):
        """
        Predict digit for one cell tensor.
        Returns (digit:int, confidence:float)
        """
        self.eval()
        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item()
