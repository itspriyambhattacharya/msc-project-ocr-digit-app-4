# OCR Sudoku Solver — MSc Project
**Stack:** Python · PyTorch · OpenCV · Flask · HTML/CSS/JS

---

## Project Structure

```
sudoku-app/
├── backend/
│   ├── app.py          ← Flask API server (main entry point)
│   ├── model.py        ← PyTorch CNN architecture (DigitCNN)
│   ├── train.py        ← Training script (MNIST + augmentation)
│   ├── ocr.py          ← OpenCV grid extraction + digit classification
│   ├── solver.py       ← Sudoku backtracking solver
│   ├── requirements.txt
│   └── model_weights/
│       └── digit_cnn.pth   ← saved after training
└── frontend/
    └── index.html      ← Complete single-file frontend
```

---

## Setup & Run

### 1. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the digit recognition model
```bash
python train.py
```
- Downloads MNIST automatically (~11 MB)
- Trains for 20 epochs with augmentation
- Saves best model to `model_weights/digit_cnn.pth`
- Expected accuracy: **~99.2%** on MNIST test set

### 3. Start the Flask server
```bash
python app.py
```
Server runs at: `http://localhost:5000`

The frontend is served automatically at `http://localhost:5000`

---

## API Endpoints

### `POST /api/ocr`
Upload a sudoku image. Returns detected digit grid.

**Request:** `multipart/form-data` with field `image`

**Response:**
```json
{
  "board": [7,2,3,0,0,0,0,4,0, ...],  // 81 integers, 0=blank
  "digit_count": 30
}
```

---

### `POST /api/solve`
Solve a sudoku board.

**Request:**
```json
{ "board": [7,2,3,0,0,0,0,4,0, ...] }
```

**Response:**
```json
{
  "solved": true,
  "board": [7,2,3,5,6,8,9,4,1, ...],
  "conflicts": [],
  "error": null
}
```

---

## How the OCR Pipeline Works

```
Image upload
    ↓
Grayscale + Adaptive Threshold (OpenCV)
    ↓
Find largest 4-corner contour (sudoku border)
    ↓
Perspective warp → 450×450 flat grid
    ↓
Split into 81 cells (50×50 each, with margin trim)
    ↓
DigitCNN (PyTorch) classifies each cell → 0..9
    ↓
Confidence thresholding (< 0.85 → blank)
    ↓
Return 9×9 grid as JSON
```

---

## How the Solver Works

Backtracking with **Minimum Remaining Values (MRV)** heuristic:
- Always picks the empty cell with the fewest valid candidates
- Dramatically reduces backtracking vs naive left-to-right scan
- Solves any valid puzzle in milliseconds

---

## Frontend Features

| Feature | Description |
|---|---|
| Image upload | Drag & drop or file picker |
| OCR fill | Detected digits auto-fill (locked, shown in darker bg) |
| Cell selection | Click cell → highlight row/col/box |
| Number input | Numpad buttons OR keyboard 1–9 |
| Arrow navigation | Arrow keys move between cells |
| Backspace | Clears selected cell |
| Solve button | Calls backend solver, animates fill |
| Conflict highlight | Red highlight on duplicate digits |
| Reset | Restore to OCR state |
| Demo mode | Works offline with sample puzzle |

---

## Improving OCR Accuracy

For better newspaper digit recognition:

1. **Fine-tune on printed digits** — collect 100–200 printed sudoku digit images per class and fine-tune the last 2 layers
2. **Cell margin tuning** — adjust `CELL_MARGIN` in `ocr.py` if grid lines bleed into cells
3. **Confidence threshold** — lower `CONFIDENCE_THRESH` (e.g. 0.75) if too many cells are marked blank
4. **Preprocessing** — try `cv2.dilate()` on thin printed digits before classification

---

## Requirements

- Python 3.9+
- CUDA GPU optional (CPU works fine for inference)
- Modern browser (Chrome, Firefox, Safari, Edge)
