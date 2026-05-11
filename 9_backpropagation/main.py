"""
Backpropagation (Multi-Layer Perceptron) — Generalized
Auto-detects classification vs regression based on target dtype/cardinality.
Uses sklearn's MLPClassifier / MLPRegressor (trained via backpropagation).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, r2_score, mean_squared_error)

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
    print(f"Task: {'Classification' if classification else 'Regression'}")

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

    # MLP with 2 hidden layers, ReLU, Adam optimizer (backpropagation)
    if classification:
        model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                              solver='adam', max_iter=500, random_state=42)
    else:
        model = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu',
                             solver='adam', max_iter=500, random_state=42)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    print(f"\nArchitecture: {X.shape[1]} → 32 → 16 → output")
    print(f"Iterations:   {model.n_iter_}")
    print(f"Final loss:   {model.loss_:.4f}")

    if classification:
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
        print(classification_report(y_test, y_pred))
    else:
        print(f"\nR² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(model.loss_curve_, color='#2196F3', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].grid(alpha=0.3)

    if classification:
        cm = confusion_matrix(y_test, y_pred)
        im = axes[1].imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=axes[1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(j, i, cm[i, j], ha='center', va='center',
                             color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix')
    else:
        axes[1].scatter(y_test, y_pred, alpha=0.6, color='#2196F3')
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        axes[1].plot(lims, lims, color='#EF5350', linestyle='--')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
        axes[1].set_title('Actual vs Predicted')
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
