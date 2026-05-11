"""
K-Means Clustering — Generalized
Uses all numeric columns. Auto-finds best K via silhouette score (K=2..10).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

    X_raw = encode_features(df)
    if X_raw.shape[1] < 2:
        raise ValueError("Need at least 2 features after encoding.")

    X = StandardScaler().fit_transform(X_raw)
    print(f"Features used: {X.shape[1]}")

    # Find best K
    k_values = list(range(2, 11))
    wcss, silhouettes = [], []
    print(f"\n  {'K':<6} {'WCSS':<14} {'Silhouette'}")
    print("  " + "-" * 36)
    for k in k_values:
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = model.fit_predict(X)
        wcss.append(model.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"  {k:<6} {model.inertia_:<14.4f} {sil:.4f}")

    best_k = k_values[int(np.argmax(silhouettes))]
    print(f"\nBest K (highest silhouette): {best_k}")

    # Final model
    model = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
    labels = model.fit_predict(X)
    print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(k_values, wcss, 'o-', color='#2196F3', linewidth=2, markersize=8)
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('WCSS')
    axes[0].set_title('Elbow Method')
    axes[0].grid(alpha=0.3)

    axes[1].bar(k_values, silhouettes, color='#4CAF50', edgecolor='black')
    axes[1].axvline(best_k, color='#EF5350', linestyle='--', label=f'Best K={best_k}')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Scores')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Cluster scatter using first 2 dims (or PCA if many)
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        X2 = PCA(n_components=2).fit_transform(X)
        xlabel, ylabel = 'PC1', 'PC2'
    else:
        X2 = X
        xlabel, ylabel = 'Feature 1', 'Feature 2'

    for c in range(best_k):
        mask = labels == c
        axes[2].scatter(X2[mask, 0], X2[mask, 1], alpha=0.6, label=f'Cluster {c}',
                        edgecolors='black', linewidth=0.3)
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    axes[2].set_title(f'Clusters (K={best_k})')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
