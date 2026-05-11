"""
Clustering — DBSCAN, Agglomerative, Divisive — Generalized
Uses all numeric columns. Compares all three by silhouette score.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CSV_PATH = "data.csv"   # <-- change this path to your dataset
N_CLUSTERS = 5          # used by Agglomerative and Divisive


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


class DivisiveClustering:
    """Top-down: recursively split largest cluster using K-Means(k=2)."""

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, X):
        n = len(X)
        clusters = [np.arange(n)]
        while len(clusters) < self.n_clusters:
            sizes = [len(c) for c in clusters]
            i = int(np.argmax(sizes))
            big = clusters.pop(i)
            if len(big) < 2:
                clusters.append(big)
                break
            km = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
            sub = km.fit_predict(X[big])
            clusters.append(big[sub == 0])
            clusters.append(big[sub == 1])
        labels = np.zeros(n, dtype=int)
        for cid, idx in enumerate(clusters):
            labels[idx] = cid
        return labels


def main():
    df = pd.read_csv(CSV_PATH).dropna()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    X_raw = encode_features(df)
    X = StandardScaler().fit_transform(X_raw)
    print(f"Features used: {X.shape[1]}")

    # DBSCAN — try a few eps values, pick best
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    best_db_score, best_db_labels, best_eps = -1, None, None
    for eps in eps_values:
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        unique = np.unique(labels[labels != -1])
        if len(unique) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
            if score > best_db_score:
                best_db_score, best_db_labels, best_eps = score, labels, eps
        except Exception:
            pass

    if best_db_labels is None:
        best_db_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
        best_eps = 0.5
        best_db_score = float('nan')

    agg_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward').fit_predict(X)
    div_labels = DivisiveClustering(n_clusters=N_CLUSTERS).fit_predict(X)

    print(f"\n{'Algorithm':<22} {'Clusters':<12} {'Silhouette'}")
    print("-" * 50)
    for name, labels in [(f'DBSCAN (eps={best_eps})', best_db_labels),
                         ('Agglomerative (Ward)', agg_labels),
                         ('Divisive', div_labels)]:
        n_c = len(np.unique(labels[labels != -1]))
        try:
            sil = silhouette_score(X, labels) if n_c >= 2 else float('nan')
        except Exception:
            sil = float('nan')
        print(f"{name:<22} {n_c:<12} {sil:.4f}")

    # Plot — PCA to 2D
    X2 = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['DBSCAN', 'Agglomerative', 'Divisive']
    all_labels = [best_db_labels, agg_labels, div_labels]
    for ax, labels, title in zip(axes, all_labels, titles):
        unique = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 0.9, len(unique)))
        for c, col in zip(unique, colors):
            mask = labels == c
            label = f'Noise' if c == -1 else f'C{c}'
            marker = 'x' if c == -1 else 'o'
            ax.scatter(X2[mask, 0], X2[mask, 1], c=[col], alpha=0.7, s=30,
                       label=label, marker=marker)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
