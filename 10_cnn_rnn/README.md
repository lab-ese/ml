# CNN + RNN (PyTorch)

- File: `main.py`
- Change paths on lines 24-25:
  - `CNN_PATH` — auto-detects:
    - **Folder** of class subfolders → 2D CNN on images
    - **CSV** with first col = label, rest = pixel values (square count) → 2D CNN on images
    - **CSV** with text column → 1D CNN on character tokens (text classification)
    - **CSV** with numeric features → 1D CNN on feature vectors
  - `RNN_CSV_PATH` — single numeric-column time series CSV

## Setup (Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn torch torchvision pillow
```

## Run (Windows)
```cmd
python main.py
```

## Linux/Mac (optional)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy matplotlib scikit-learn torch torchvision pillow
python main.py
```
