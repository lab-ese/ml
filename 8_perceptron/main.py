"""
Single-Layer Perceptron — Generalized
Auto-detects: all columns except last as features, last column as binary class.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def main():
    df = pd.read_csv(CSV_PATH).dropna()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    y_col = df.columns[-1]
    y_raw = df[y_col].values
    X = encode_features(df.drop(columns=[y_col]))

    if y_raw.dtype == object:
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw

    n_classes = len(np.unique(y))
    print(f"Features: {X.shape[1]}  |  Classes: {n_classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Compare learning rates (eta0)
    lrs = [0.001, 0.01, 0.1, 0.5, 1.0]
    accs = []
    print(f"\n  {'LR':<8} {'Accuracy'}")
    print("  " + "-" * 22)
    for lr in lrs:
        model = Perceptron(eta0=lr, max_iter=200, random_state=42)
        model.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        accs.append(acc)
        print(f"  {lr:<8} {acc*100:.2f}%")

    best_lr = lrs[int(np.argmax(accs))]
    print(f"\nBest LR: {best_lr}")

    model = Perceptron(eta0=best_lr, max_iter=200, random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    print(f"\nFinal Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar([str(lr) for lr in lrs], [a*100 for a in accs],
                color='#2196F3', edgecolor='black')
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Learning Rate Comparison')
    axes[0].grid(axis='y', alpha=0.3)

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

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
