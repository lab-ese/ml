import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Dataset
df = pd.DataFrame({'SepalLength': [5.1,4.9,4.7,4.6,5.0,5.4,4.6,5.0,4.4,4.9,7.0,6.4,6.9,5.5,6.5],
                   'SepalWidth': [3.5,3.0,3.2,3.1,3.6,3.9,3.4,3.4,2.9,3.1,3.2,3.2,3.1,2.8,3.2],
                   'PetalLength': [1.4,1.4,1.3,1.5,1.4,1.7,1.4,1.5,1.4,1.5,4.7,4.5,4.9,4.0,5.0],
                   'PetalWidth': [0.2,0.2,0.2,0.2,0.2,0.4,0.3,0.2,0.2,0.2,1.4,1.5,1.5,1.3,2.0],
                   'Species': [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]})
print("Data:\n", df)

X, y = df.drop('Species',axis=1), df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"\nDecision Tree Accuracy: {dt_acc:.4f}")

# Visualize
plt.figure(figsize=(16,8)); plot_tree(dt, feature_names=X.columns, class_names=['Setosa','Versicolor'], filled=True, rounded=True); plt.savefig('decision_tree.png'); plt.close()

# Random Forest
rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# Feature importance
plt.figure(figsize=(10,5)); x = np.arange(len(X.columns)); plt.bar(x-0.15, dt.feature_importances_, 0.3, label='DT'); plt.bar(x+0.15, rf.feature_importances_, 0.3, label='RF'); plt.xticks(x, X.columns, rotation=45); plt.title('Feature Importance'); plt.legend(); plt.savefig('feature_importance.png'); plt.close()

print(f"\nDT Report:\n{classification_report(y_test, dt.predict(X_test), target_names=['Setosa','Versicolor'])}")
print(f"RF Report:\n{classification_report(y_test, rf.predict(X_test), target_names=['Setosa','Versicolor'])}")
print("\nDone!")