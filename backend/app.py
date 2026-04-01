"""
app.py
======
LOCATION : cnn_ocr_app_msc_project_4\\backend\\app.py
HOW TO RUN (from VS Code terminal):

    cd D:\\coding\\projects\\cnn_ocr_app_msc_project_4\\backend
    python app.py

Then open: http://127.0.0.1:5000
"""

from solver import solve_sudoku
from ocr import extract_sudoku_from_image, load_model
from flask_cors import CORS
from flask import Flask, request, jsonify, send_from_directory
import os
import sys

# ── Resolve absolute paths from this file's location ─────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))        # …\backend
# …\cnn_ocr_app_msc_project_4
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
MODEL_PATH = os.path.join(BACKEND_DIR, "model_weights", "digit_cnn.pth")

# Add backend\ to sys.path so plain imports (ocr, solver, model) always resolve
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ── Startup info ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  OCR Sudoku Solver — Starting")
print("=" * 55)
print(f"  Backend  : {BACKEND_DIR}")
print(f"  Frontend : {FRONTEND_DIR}")
print(f"  Model    : {MODEL_PATH}")
idx_ok = os.path.isfile(os.path.join(FRONTEND_DIR, "index.html"))
print(
    f"  index.html found : {'YES' if idx_ok else 'NO  ← check frontend folder!'}")
print("=" * 55 + "\n")

# ── Load PyTorch model (optional) ─────────────────────────────────────────────
model = None
MODEL_READY = False

if os.path.isfile(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        MODEL_READY = True
        print("  OCR mode : PyTorch CNN  (model loaded)\n")
    except Exception as e:
        print(f"  WARNING  : Could not load model weights — {e}")
        print("  OCR mode : kNN fallback\n")
else:
    print("  OCR mode : kNN fallback  (run train.py to enable CNN)\n")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/ocr", methods=["POST"])
def ocr_endpoint():
    """
    Upload a sudoku image → detect grid → classify digits.
    Request  : multipart/form-data,  field name = 'image'
    Response : { board:[81 ints], digit_count:int, mode:str }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image received. "
                                 "Send multipart/form-data with field name 'image'."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file received."}), 400

    allowed = {"image/jpeg", "image/jpg",
               "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        return jsonify({"error": f"File type '{file.content_type}' not supported. "
                        "Please upload a JPEG or PNG image."}), 400

    try:
        img_bytes = file.read()
        digits = extract_sudoku_from_image(
            img_bytes,
            model=model if MODEL_READY else None)

        return jsonify({
            "board":       digits,
            "digit_count": sum(1 for d in digits if d != 0),
            "mode":        "cnn" if MODEL_READY else "knn_fallback"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        app.logger.exception("Unexpected OCR error")
        return jsonify({"error": f"OCR failed: {str(e)}. "
                        "Try a clearer, well-lit image."}), 500


@app.route("/api/solve", methods=["POST"])
def solve_endpoint():
    """
    Solve a sudoku puzzle.
    Request  : JSON  { "board": [81 ints] }   (0 = blank cell)
    Response : { solved:bool, board:[81 ints], conflicts:[[i,j],...], error:str|null }
    """
    data = request.get_json(silent=True)
    if not data or "board" not in data:
        return jsonify({"error": "Send JSON with key 'board'."}), 400

    board = data["board"]
    if not isinstance(board, list) or len(board) != 81:
        return jsonify({"error": "'board' must be a list of exactly 81 integers."}), 400
    if not all(isinstance(v, int) and 0 <= v <= 9 for v in board):
        return jsonify({"error": "Every value must be an integer 0–9 (0 = blank)."}), 400

    return jsonify(solve_sudoku(board))


@app.route("/api/health")
def health():
    """Health check — confirms server is up and reports OCR mode."""
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_READY,
        "ocr_mode":    "cnn" if MODEL_READY else "knn_fallback",
        "index_found": os.path.isfile(os.path.join(FRONTEND_DIR, "index.html"))
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"  Open in browser: http://127.0.0.1:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
