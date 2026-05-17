import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Dataset 1: Location
df1 = pd.DataFrame({'Latitude': [40.70,40.80,40.90,41.00,41.10,40.50,40.60,41.20,41.30,41.40,40.71,40.72,41.31,41.32],
                    'Longitude': [-74.00,-74.10,-74.20,-74.00,-74.10,-73.80,-73.90,-74.20,-74.30,-74.10,-73.91,-73.92,-74.21,-74.22]})
# Dataset 2: Travel
df2 = pd.DataFrame({'Distance_km': [10,20,30,80,90,100,40,50,60,70,15,25,85,95],
                    'Duration_min': [15,25,35,90,100,110,45,55,65,75,20,30,95,105]})

def compare(df, name):
    X, Xs = df.values, StandardScaler().fit_transform(df.values)
    dbs, km = DBSCAN(eps=0.5, min_samples=2).fit_predict(Xs), KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(Xs)
    ds = silhouette_score(Xs[dbs>=0], dbs[dbs>=0]) if len(set(dbs))>1 else -1
    ks = silhouette_score(Xs, km)
    print(f"{name}: DBSCAN sil={ds:.4f}, K-means sil={ks:.4f}, DBSCAN clusters={len(set(dbs))-(1 if -1 in dbs else 0)}, noise={sum(dbs==-1)}")
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    axes[0].scatter(df.iloc[:,0], df.iloc[:,1], c='blue', s=100); axes[0].set_title('Original')
    axes[1].scatter(df.iloc[:,0], df.iloc[:,1], c=dbs, cmap='viridis', s=100); axes[1].set_title(f'DBSCAN')
    axes[2].scatter(df.iloc[:,0], df.iloc[:,1], c=km, cmap='viridis', s=100); axes[2].scatter(KMeans(n_clusters=2,random_state=42,n_init=10).fit(Xs).cluster_centers_[:,0],KMeans(n_clusters=2,random_state=42,n_init=10).fit(Xs).cluster_centers_[:,1],c='red',marker='X',s=200); axes[2].set_title('K-means')
    plt.savefig(f'/Users/sapatmohit18/Desktop/ML/dbscan_{name.lower()}.png'); plt.close()

compare(df1, 'Location')
compare(df2, 'Travel')

print("\nParameter tuning (Location):")
for eps in [0.3,0.5,0.7,1.0]:
    labels = DBSCAN(eps=eps, min_samples=2).fit_predict(StandardScaler().fit_transform(df1.values))
    print(f"eps={eps}: {len(set(labels))-(1 if -1 in labels else 0)} clusters, {sum(labels==-1)} noise")
print("\nDone!")