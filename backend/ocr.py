"""
backend/ocr.py
==============
Sudoku OCR Pipeline — final robust version.

Key principle: "A missed digit is acceptable. A wrong digit is NEVER acceptable."

This version implements a strict two-gate rejection policy:
  Gate 1 — Geometric validation (before CNN):
    The digit contour must pass shape checks. A real digit has a minimum
    width AND height. Grid-line remnants are very thin (aspect ratio > 4:1
    in the narrow direction) and are rejected here.

  Gate 2 — High-confidence requirement for ambiguous predictions:
    The digit '1' is the most common false positive (grid lines look like 1s).
    Class-specific confidence thresholds — '1' requires 0.92, others 0.65.
    Any prediction where the 2nd-best class has > 30% probability is rejected
    (means the model is genuinely unsure → return blank).

Other fixes:
  - Multi-strategy binarisation: try 4 methods, pick best via contour score.
  - Border-region polarity fix: handles coloured backgrounds (blue/grey cells).
  - Digit centering on 28×28 canvas before CNN (aligns with MNIST training).
  - MIN_DIGIT_AREA_RATIO = 0.02, MIN_DIGIT_DIM = 4 (catches small digits).
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BACKEND_DIR)
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")

# ── Config ─────────────────────────────────────────────────────────────────────
GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 9   # 50 px
CELL_MARGIN = 6
MIN_DIGIT_AREA_RATIO = 0.02             # contour must cover ≥2% of cell area
MIN_DIGIT_DIM = 4               # bounding box must be ≥4 px in both axes
MAX_ASPECT_RATIO = 3.5             # bounding box w/h or h/w must be < 3.5
# (rejects thin grid-line artifacts)

# ── Class-specific confidence thresholds ──────────────────────────────────────
# '1' is the most common false-positive (grid lines look like 1s).
# All other digits use a lower general threshold.
CONF_THRESH_CLASS_1 = 0.92   # very strict for class 1
CONF_THRESH_GENERAL = 0.65   # general threshold for classes 2-9
# If 2nd-best class probability exceeds this, the model is unsure → blank
MAX_SECOND_BEST_PROB = 0.30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(path):
    from model import DigitCNN
    m = DigitCNN().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — DECODE & UPSCALE
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Cannot decode image. Upload a JPEG, PNG, or BMP file.")
    h, w = img.shape[:2]
    if max(h, w) < 500:
        scale = 500 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — GRID DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _threshold_variants(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    t1 = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    t2 = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,    cv2.THRESH_BINARY_INV, 15, 3)
    _, t3 = cv2.threshold(blurred, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — PERSPECTIVE WARP
# ═══════════════════════════════════════════════════════════════════════════════

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
    dst = np.array([[0, 0], [GRID_SIZE-1, 0],
                    [GRID_SIZE-1, GRID_SIZE-1], [0, GRID_SIZE-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (GRID_SIZE, GRID_SIZE))


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — CELL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MULTI-STRATEGY CELL BINARISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _binarise_single(cell, method):
    """Binarise using one strategy. Always returns digit=WHITE, bg=BLACK."""
    blurred = cv2.GaussianBlur(cell, (3, 3), 0)

    if method == 'adaptive_c2':
        binary = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
    elif method == 'adaptive_c4':
        binary = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    elif method == 'otsu':
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'otsu_inv':
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = np.zeros_like(blurred)

    # Polarity fix: check outermost 3-px border (always background, never digit)
    border = np.zeros_like(binary, dtype=bool)
    border[:3, :] = True
    border[-3:, :] = True
    border[:, :3] = True
    border[:, -3:] = True
    if np.mean(binary[border]) > 128:      # border is mostly white → inverted
        binary = cv2.bitwise_not(binary)

    return binary   # digit=WHITE, bg=BLACK


def _score_binarisation(binary):
    """Score how clearly a binarisation shows a digit contour (higher=better)."""
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(binary, kernel, iterations=1)
    cnts, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cell_area = binary.shape[0] * binary.shape[1]
    valid = [c for c in cnts if cv2.contourArea(
        c) >= cell_area * MIN_DIGIT_AREA_RATIO]
    if not valid:
        return 0.0
    return min(cv2.contourArea(max(valid, key=cv2.contourArea)) / cell_area, 0.5)


def _binarise_cell(cell):
    """Try 4 strategies, return the best binarisation."""
    best_bin, best_score = None, -1.0
    for m in ['adaptive_c2', 'adaptive_c4', 'otsu', 'otsu_inv']:
        try:
            b = _binarise_single(cell, m)
            s = _score_binarisation(b)
            if s > best_score:
                best_score, best_bin = s, b
        except Exception:
            continue
    if best_bin is None:
        _, best_bin = cv2.threshold(cell, 127, 255, cv2.THRESH_BINARY_INV)
        if np.mean(best_bin) > 128:
            best_bin = cv2.bitwise_not(best_bin)
    return best_bin


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — BLANK DETECTION + GEOMETRIC VALIDATION  (GATE 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_digit_contour(binary):
    """
    Find the largest digit-like contour.
    Returns (contour, bounding_rect) or (None, None) if cell is blank.

    Gate 1 geometry checks:
      - Contour area >= MIN_DIGIT_AREA_RATIO * cell_area
      - Bounding box w >= MIN_DIGIT_DIM and h >= MIN_DIGIT_DIM
      - Aspect ratio (w/h or h/w) < MAX_ASPECT_RATIO
        → rejects grid-line remnants that are extremely thin
    """
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.erode(binary, kernel, iterations=1)
    cnts, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    cell_area = binary.shape[0] * binary.shape[1]

    # Filter by area first
    valid = [c for c in cnts
             if cv2.contourArea(c) >= cell_area * MIN_DIGIT_AREA_RATIO]
    if not valid:
        return None, None

    # Merge all valid contours to get the overall bounding box
    # (handles multi-part digits like 8, or digits split by noise)
    best = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)

    # Gate 1a: minimum dimension
    if w < MIN_DIGIT_DIM or h < MIN_DIGIT_DIM:
        return None, None

    # Gate 1b: aspect ratio — rejects grid-line artifacts
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        return None, None

    return best, (x, y, w, h)


def _is_blank(cell):
    binary = _binarise_cell(cell)
    cnt, _ = _get_digit_contour(binary)
    return cnt is None


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — DIGIT CENTERING FOR CNN
# ═══════════════════════════════════════════════════════════════════════════════

def _prepare_digit_for_cnn(cell):
    """
    Crop digit bounding box + 20% padding → scale to 20×20 →
    centre on 28×28 black canvas → return BLACK digit on WHITE bg (MNIST style).
    """
    binary = _binarise_cell(cell)
    cnt, bbox = _get_digit_contour(binary)

    if cnt is None or bbox is None:
        resized = cv2.resize(binary, (20, 20))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        canvas[4:24, 4:24] = resized
        return cv2.bitwise_not(canvas)

    x, y, w, h = bbox
    px = max(2, int(w * 0.20))
    py = max(2, int(h * 0.20))
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(binary.shape[1], x + w + px)
    y2 = min(binary.shape[0], y + h + py)

    crop = binary[y1:y2, x1:x2]
    if crop.size == 0:
        crop = binary

    dh, dw = crop.shape
    scale = 20.0 / max(dh, dw)
    nw = max(1, int(dw * scale))
    nh = max(1, int(dh * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - nh) // 2
    left = (28 - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized

    return cv2.bitwise_not(canvas)   # BLACK digit on WHITE bg


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8A — kNN CLASSIFIER (no trained model)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_features(img_28x28):
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
        ds = datasets.MNIST(_DATA_DIR, train=True, download=True,
                            transform=transforms.ToTensor())
        X, y = [], []
        counts = {i: 0 for i in range(1, 10)}
        for img_t, label in ds:
            if label == 0 or counts.get(label, 0) >= 600:
                continue
            img_np = (img_t.numpy()[0] * 255).astype(np.uint8)
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
        white_on_black = cv2.bitwise_not(prepared)
        digit, conf = knn.predict(_extract_features(white_on_black))
        thresh = CONF_THRESH_CLASS_1 if digit == 1 else CONF_THRESH_GENERAL
        # kNN confs are smaller
        digits.append(digit if conf >= thresh * 0.4 else 0)
    return digits


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8B — CNN CLASSIFIER WITH GATE 2 REJECTION  (trained DigitCNN)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_confident_prediction(pred, probs_row):
    """
    Gate 2: reject predictions that are wrong-looking even if CNN says so.

    Rules (all must pass):
      1. Prediction is not class 0 (blank)
      2. For class 1: confidence >= CONF_THRESH_CLASS_1 (0.92)
         For others:  confidence >= CONF_THRESH_GENERAL  (0.65)
      3. 2nd-best class probability < MAX_SECOND_BEST_PROB (0.30)
         — if the model is split between two digits, return blank
    """
    if pred == 0:
        return False

    best_conf = probs_row[pred].item()

    # Rule 2: class-specific threshold
    thresh = CONF_THRESH_CLASS_1 if pred == 1 else CONF_THRESH_GENERAL
    if best_conf < thresh:
        return False

    # Rule 3: check 2nd-best probability
    sorted_probs = sorted(probs_row.tolist(), reverse=True)
    if len(sorted_probs) > 1 and sorted_probs[1] > MAX_SECOND_BEST_PROB:
        return False

    return True


def _classify_cnn(cells, model):
    blank_flags = [_is_blank(c) for c in cells]
    active_idx = [i for i, b in enumerate(blank_flags) if not b]
    if not active_idx:
        return [0] * 81

    tensors = [_cnn_transform(_prepare_digit_for_cnn(cells[i]))
               for i in active_idx]
    batch = torch.stack(tensors).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(batch), dim=1)        # (N, 10)
        preds = probs.argmax(dim=1)

    results = [0] * 81
    for j, i in enumerate(active_idx):
        pred = preds[j].item()
        prob_row = probs[j]
        if _is_confident_prediction(pred, prob_row):
            results[i] = pred
        # else: leave as 0 (blank) — wrong detection is never acceptable

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sudoku_from_image(img_bytes, model=None):
    """
    Raw image bytes → list of 81 ints (0=blank, 1-9=digit).
    model=None → kNN fallback. Pass DigitCNN for CNN mode.
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
