"""
Classification — Decision Tree, Naive Bayes, KNN — Generalized
Auto-detects: all columns except last as features, last column as target (class).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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

    # Last column = target
    y_col = df.columns[-1]
    y_raw = df[y_col].values
    X = encode_features(df.drop(columns=[y_col]))

    # Encode target if string
    if y_raw.dtype == object:
        y = LabelEncoder().fit_transform(y_raw)
    else:
        y = y_raw

    print(f"\nFeatures: {X.shape[1]}  |  Classes: {len(np.unique(y))}")
    print(f"Target:   {y_col}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes':   GaussianNB(),
        'KNN (k=5)':     KNeighborsClassifier(n_neighbors=5),
    }

    accs = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        accs[name] = acc
        print(f"\n{'='*50}")
        print(f"  {name}  —  Accuracy: {acc*100:.2f}%")
        print('='*50)
        print(classification_report(y_test, y_pred))

    # Comparison plot
    plt.figure(figsize=(8, 5))
    names = list(accs.keys())
    vals = [accs[n] * 100 for n in names]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    bars = plt.bar(names, vals, color=colors, edgecolor='black')
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{v:.2f}%', ha='center', fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.title('Classifier Comparison')
    plt.ylim(0, 105)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
