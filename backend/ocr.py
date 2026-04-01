"""
backend/ocr.py
==============
Sudoku OCR Pipeline — robust printed-digit recognition.

Root cause of previous misdetections:
  1. _normalize_cell() used Otsu on whole cell — for thin printed digits
     this binarised incorrectly, making digits look like noise.
  2. BLANK_RATIO pixel-density check was too fragile for thin printed strokes.
  3. Cells were NOT centred before passing to CNN, so the model
     (trained on centred MNIST digits) scored very poorly.
  4. Confidence threshold 0.82 was too aggressive for printed fonts.

Fixes:
  - Adaptive threshold per cell (not global Otsu).
  - Contour-based blank detection (finds actual digit shape).
  - Digit centering: crop bounding box + padding, place on 28x28 canvas.
  - Confidence threshold 0.55 for printed fonts.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BACKEND_DIR)
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")

GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9
CELL_MARGIN = 6
CONFIDENCE_THRESH = 0.55
MIN_DIGIT_AREA_RATIO = 0.04
MIN_DIGIT_DIM = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(path):
    from model import DigitCNN
    m = DigitCNN().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


# ── Step 1: Decode ─────────────────────────────────────────────────────────────

def _decode_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Upload a JPEG, PNG, or BMP.")
    h, w = img.shape[:2]
    if max(h, w) < 500:
        scale = 500 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


# ── Step 2: Grid detection ─────────────────────────────────────────────────────

def _threshold_variants(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    t1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    t2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 3)
    _, t3 = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return t1, t2, t3


def _find_quad(thresh):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, k, iterations=1)
    cnts, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    img_area = thresh.shape[0] * thresh.shape[1]
    for cnt in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(cnt) < img_area * 0.15:
            break
        peri = cv2.arcLength(cnt, True)
        for eps in [0.01, 0.02, 0.03, 0.05, 0.08]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _find_grid_corners(gray):
    for thresh in _threshold_variants(gray):
        quad = _find_quad(thresh)
        if quad is not None:
            return quad
    h, w = gray.shape
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)


# ── Step 3: Warp ───────────────────────────────────────────────────────────────

def _order_corners(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _warp_to_square(gray, corners):
    rect = _order_corners(corners)
    dst = np.array([[0, 0], [GRID_SIZE-1, 0], [GRID_SIZE-1,
                   GRID_SIZE-1], [0, GRID_SIZE-1]], dtype=np.float32)
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(rect, dst), (GRID_SIZE, GRID_SIZE))


# ── Step 4: Cell extraction ────────────────────────────────────────────────────

def _extract_cells(warped):
    cells = []
    for row in range(9):
        for col in range(9):
            y1 = row * CELL_SIZE + CELL_MARGIN
            y2 = (row + 1) * CELL_SIZE - CELL_MARGIN
            x1 = col * CELL_SIZE + CELL_MARGIN
            x2 = (col + 1) * CELL_SIZE - CELL_MARGIN
            cells.append(warped[y1:y2, x1:x2])
    return cells


# ── Step 5: Cell binarisation (KEY FIX) ───────────────────────────────────────

def _binarise_cell(cell):
    """
    Adaptive threshold per cell — handles local contrast variation from
    newspaper printing, shadows, and JPEG compression artifacts.
    Returns: digit=WHITE, background=BLACK
    """
    blurred = cv2.GaussianBlur(cell, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=4)
    # If most pixels are white, background is dark → invert
    if np.mean(binary) > 128:
        binary = cv2.bitwise_not(binary)
    return binary  # digit=WHITE, bg=BLACK


# ── Step 6: Blank detection (KEY FIX) ─────────────────────────────────────────

def _get_digit_contour(binary):
    """
    Contour-based blank detection. Much more robust than pixel ratio
    for thin printed digits which have low pixel density but clear shape.
    """
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(binary, kernel, iterations=1)  # remove grid-line noise
    cnts, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cell_area = binary.shape[0] * binary.shape[1]
    valid = [c for c in cnts if cv2.contourArea(
        c) >= cell_area * MIN_DIGIT_AREA_RATIO]
    if not valid:
        return None
    return max(valid, key=cv2.contourArea)


def _is_blank(cell):
    binary = _binarise_cell(cell)
    cnt = _get_digit_contour(binary)
    if cnt is None:
        return True
    _, _, w, h = cv2.boundingRect(cnt)
    return w < MIN_DIGIT_DIM or h < MIN_DIGIT_DIM


# ── Step 7: Digit centering for CNN (KEY FIX) ─────────────────────────────────

def _prepare_digit_for_cnn(cell):
    """
    Crop the digit bounding box, add padding, centre on 28x28 canvas.
    This is CRITICAL — CNN trained on centred MNIST digits fails badly
    on un-centred raw cell crops. This fix aligns the formats.

    Returns: BLACK digit on WHITE background (28x28) — MNIST convention.
    """
    binary = _binarise_cell(cell)
    cnt = _get_digit_contour(binary)

    if cnt is None:
        # Fallback: use whole cell
        resized = cv2.resize(binary, (20, 20))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = resized
        return cv2.bitwise_not(canvas)

    x, y, w, h = cv2.boundingRect(cnt)

    # 20% padding around digit bounding box
    px = max(2, int(w * 0.20))
    py = max(2, int(h * 0.20))
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(binary.shape[1], x + w + px)
    y2 = min(binary.shape[0], y + h + py)

    crop = binary[y1:y2, x1:x2]
    if crop.size == 0:
        crop = binary

    # Scale to fit 20x20 preserving aspect ratio
    dh, dw = crop.shape
    scale = 20.0 / max(dh, dw)
    nw, nh = max(1, int(dw*scale)), max(1, int(dh*scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    # Centre on 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - nh) // 2
    left = (28 - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized

    return cv2.bitwise_not(canvas)  # BLACK digit on WHITE bg (MNIST style)


# ── Step 8A: kNN classifier (no model) ───────────────────────────────────────

def _extract_features(img_28x28):
    """16-dim 4x4 block density features from a 28x28 digit image."""
    return np.array([
        np.mean(img_28x28[br*7:(br+1)*7, bc*7:(bc+1)*7]) / 255.0
        for br in range(4) for bc in range(4)
    ], dtype=np.float32)


class _KNNClassifier:
    def __init__(self): self.X = self.y = None
    def fit(self, X, y): self.X = np.array(X, np.float32); self.y = np.array(y)

    def predict(self, x, k=3):
        if self.X is None:
            return 0, 0.0
        dists = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(dists)[:k]
        lbls, cnts = np.unique(self.y[idx], return_counts=True)
        best = lbls[np.argmax(cnts)]
        conf = float(np.max(cnts)) / k / (1.0 + dists[idx[0]])
        return int(best), conf


_knn_clf = None


def _get_knn():
    global _knn_clf
    if _knn_clf is not None:
        return _knn_clf
    _knn_clf = _KNNClassifier()
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        from torchvision import datasets
        ds = datasets.MNIST(_DATA_DIR, train=True,
                            download=True, transform=transforms.ToTensor())
        X, y = [], []
        counts = {i: 0 for i in range(1, 10)}
        for img_t, label in ds:
            if label == 0 or counts.get(label, 0) >= 600:
                continue
            img_np = (img_t.numpy()[0] * 255).astype(np.uint8)
            # WHITE digit on BLACK (MNIST native) → invert for our feature extractor
            X.append(_extract_features(img_np))
            y.append(label)
            counts[label] += 1
            if all(v >= 600 for v in counts.values()):
                break
        if X:
            _knn_clf.fit(X, y)
            print(f"  kNN ready — {len(X)} samples.")
    except Exception as e:
        print(f"  kNN warning: {e}")
    return _knn_clf


def _classify_knn(cells):
    knn = _get_knn()
    digits = []
    for cell in cells:
        if _is_blank(cell):
            digits.append(0)
            continue
        prepared = _prepare_digit_for_cnn(cell)
        # back to WHITE digit for features
        white_on_black = cv2.bitwise_not(prepared)
        digit, conf = knn.predict(_extract_features(white_on_black))
        digits.append(digit if conf >= 0.18 else 0)
    return digits


# ── Step 8B: CNN classifier (trained model) ────────────────────────────────────

def _classify_cnn(cells, model):
    blank_flags = [_is_blank(c) for c in cells]
    active_idx = [i for i, b in enumerate(blank_flags) if not b]
    if not active_idx:
        return [0] * 81

    tensors = [_cnn_transform(_prepare_digit_for_cnn(cells[i]))
               for i in active_idx]
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)
        confs, preds = probs.max(dim=1)

    results = [0] * 81
    for j, i in enumerate(active_idx):
        pred, conf = preds[j].item(), confs[j].item()
        results[i] = 0 if (pred == 0 or conf < CONFIDENCE_THRESH) else pred
    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_sudoku_from_image(img_bytes, model=None):
    """
    Raw image bytes → list of 81 ints (0=blank, 1-9=digit).
    model=None uses kNN fallback; pass loaded DigitCNN for CNN mode.
    """
    img = _decode_image(img_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = _find_grid_corners(gray)
    warped = _warp_to_square(gray, corners)
    cells = _extract_cells(warped)
    return _classify_cnn(cells, model) if model is not None else _classify_knn(cells)


def print_grid(digits):
    for r in range(9):
        row = digits[r*9:(r+1)*9]
        if r in (3, 6):
            print("──────┼───────┼──────")
        parts = []
        for c, v in enumerate(row):
            if c in (3, 6):
                parts.append("│")
            parts.append(str(v) if v else "·")
        print(" ".join(parts))
