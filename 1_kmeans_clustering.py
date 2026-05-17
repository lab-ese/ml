import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dataset
df = pd.DataFrame({'StudyHours': [2,3,4,5,6,7,8,9,1,2.5,4.5,6.5,8.5,1.5,3.5],
                   'TestScore': [50,55,60,65,70,75,80,85,45,52,62,72,82,48,58]})
print("Data:\n", df)

X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(1,7)]
plt.figure(figsize=(8,4)); plt.plot(range(1,7), inertias, 'bo-'); plt.xlabel('k'); plt.ylabel('Inertia'); plt.title('Elbow'); plt.savefig('/Users/sapatmohit18/Desktop/ML/kmeans_elbow.png'); plt.close()

# K-means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df['Cluster'] = kmeans.labels_
print("\nClusters:", df['Cluster'].value_counts().to_dict())

plt.figure(figsize=(8,6))
for i in range(3): plt.scatter(X[kmeans.labels_==i,0], X[kmeans.labels_==i,1], s=100, label=f'Cluster {i}')
plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:,0], scaler.inverse_transform(kmeans.cluster_centers_)[:,1], c='black', marker='X', s=200, label='Centroids')
plt.xlabel('StudyHours'); plt.ylabel('TestScore'); plt.title('K-means Clusters'); plt.legend(); plt.savefig('/Users/sapatmohit18/Desktop/ML/kmeans_clusters.png'); plt.close()
print("\nDone!")