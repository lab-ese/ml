"""
K-Nearest Neighbors — Generalized
Auto-detects: all columns except last as features, last column as target (class).
Tests multiple K values and picks best.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
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

    print(f"\nFeatures: {X.shape[1]}  |  Classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Try multiple K
    k_values = [1, 3, 5, 7, 9, 11, 15, 21]
    accs = []
    print(f"\n  {'K':<6} {'Accuracy'}")
    print("  " + "-" * 20)
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_s))
        accs.append(acc)
        print(f"  {k:<6} {acc*100:.2f}%")

    best_k = k_values[int(np.argmax(accs))]
    best_acc = max(accs)
    print(f"\nBest K: {best_k}  |  Accuracy: {best_acc*100:.2f}%")

    # Final model
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print("\n" + classification_report(y_test, y_pred))

    # Plot accuracy vs K
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(k_values, [a*100 for a in accs], 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[0].axvline(best_k, color='#EF5350', linestyle='--', label=f'Best K={best_k}')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('K vs Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    cm = confusion_matrix(y_test, y_pred)
    im = axes[1].imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=axes[1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title(f'Confusion Matrix (K={best_k})')

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
