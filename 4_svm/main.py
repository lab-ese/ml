"""
Support Vector Machines — Classification & Regression — Generalized
Auto-detects whether target is categorical (classification) or continuous (regression).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)

CSV_PATH = "data.csv"   # <-- change this path to your dataset


def encode_features(df_features):
    """Auto-handle numeric / categorical / free-text columns -> dense float matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    parts = []
    for col in df_features.columns:
        s = df_features[col]
        if s.dtype == object:
            avg_len = s.astype(str).str.len().mean()
            if avg_len > 20 or s.nunique() > 50:
                vec = TfidfVectorizer(max_features=200, stop_words='english')
                parts.append(vec.fit_transform(s.astype(str)).toarray())
            else:
                parts.append(pd.get_dummies(s, drop_first=True).values.astype(float))
        else:
            parts.append(s.values.reshape(-1, 1).astype(float))
    return np.hstack(parts).astype(np.float32) if parts else np.zeros((len(df_features), 0))


def is_classification(y):
    """Heuristic: if dtype is object OR few unique values, treat as classification."""
    if y.dtype == object:
        return True
    return len(np.unique(y)) <= 20 and np.all(y == y.astype(int))


def main():
    df = pd.read_csv(CSV_PATH).dropna()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    y_col = df.columns[-1]
    y_raw = df[y_col].values
    X = encode_features(df.drop(columns=[y_col]))

    classification = is_classification(y_raw)
    print(f"\nTarget: {y_col}  |  Task: {'Classification' if classification else 'Regression'}")

    if classification:
        y = LabelEncoder().fit_transform(y_raw) if y_raw.dtype == object else y_raw
        stratify = y
    else:
        y = y_raw.astype(float)
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if classification:
        kernels = ['linear', 'rbf', 'poly']
        accs = {}
        for k in kernels:
            model = SVC(kernel=k, random_state=42)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            accs[k] = acc
            print(f"\n=== Kernel: {k} ===  Accuracy: {acc*100:.2f}%")
            print(classification_report(y_test, y_pred))

        plt.figure(figsize=(8, 5))
        bars = plt.bar(list(accs.keys()), [v*100 for v in accs.values()],
                       color=['#2196F3', '#4CAF50', '#FF9800'], edgecolor='black')
        for bar, v in zip(bars, accs.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{v*100:.2f}%', ha='center', fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.title('SVM Classification — Kernel Comparison')
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
    else:
        kernels = ['linear', 'rbf', 'poly']
        scores = {}
        for k in kernels:
            model = SVR(kernel=k)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            scores[k] = (r2, rmse)
            print(f"\n=== Kernel: {k} ===  R²: {r2:.4f}  |  RMSE: {rmse:.4f}")

        # Plot last model's predictions
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, color='#2196F3', alpha=0.6)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        plt.plot(lims, lims, color='#EF5350', linewidth=2, linestyle='--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'SVM Regression (kernel={k}) — Actual vs Predicted')
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
