# Classification — Decision Tree, Naive Bayes, KNN

- File: `main.py`
- Change CSV path on line 14: `CSV_PATH = "data.csv"`
- Last column = class target. Other columns auto-handled:
  numeric → kept, short categorical → one-hot, free text → TF-IDF.

## Setup (Windows)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib
```

## Run (Windows)
```cmd
python main.py
```

## Linux/Mac (optional)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib
python main.py
```
