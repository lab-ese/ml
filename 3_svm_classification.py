import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset
df = pd.DataFrame({'Age': [25,30,35,40,45,50,55,60,22,28,32,38,48,52,42],
                   'Income': [30,35,40,50,60,70,80,90,25,32,38,45,55,65,50],
                   'Purchased': [0,0,0,1,1,1,1,1,0,0,0,0,1,1,1]})
print("Data:\n", df)

X, y = df[['Age','Income']], df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

# Train different kernels
for kernel in ['linear','rbf','poly']:
    acc = accuracy_score(y_test, SVC(kernel=kernel, random_state=42).fit(X_train_s,y_train).predict(X_test_s))
    print(f"{kernel}: {acc:.4f}")

# Best model
best = SVC(kernel='rbf', random_state=42).fit(X_train_s, y_train)
y_pred = best.predict(X_test_s)
print(f"\nClassification Report:\n{classification_report(y_test,y_pred,target_names=['NotPurchased','Purchased'])}")

# Decision boundary
h, x_min, x_max = 0.02, X_test_s[:,0].min()-1, X_test_s[:,1].min()-1
xx, yy = np.meshgrid(np.arange(x_min, X_test_s[:,0].max()+1, h), np.arange(x_min, X_test_s[:,1].max()+1, h))
plt.figure(figsize=(10,6)); plt.contourf(xx,yy,best.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape), alpha=0.3, cmap='coolwarm')
plt.scatter(X_test_s[:,0], X_test_s[:,1], c=y_test, cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('Age (scaled)'); plt.ylabel('Income (scaled)'); plt.title('SVM Decision Boundary'); plt.savefig('/Users/sapatmohit18/Desktop/ML/svm_rbf_boundary.png'); plt.close()
print("\nDone!")