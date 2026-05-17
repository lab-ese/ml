import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
n = 60

# Dataset: Students pass/fail - clear separation
pass_hours = np.random.normal(7, 1, n)
fail_hours = np.random.normal(2, 1, n)
pass_att = np.random.normal(85, 5, n)
fail_att = np.random.normal(50, 8, n)

data = {
    'StudyHours': np.concatenate([pass_hours, fail_hours]).clip(0, 12),
    'Attendance': np.concatenate([pass_att, fail_att]).clip(30, 100),
    'Passed': [1]*n + [0]*n
}
df = pd.DataFrame(data)
print("Data:\n", df.head())

X, y = df.drop('Passed', axis=1), df['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=500, random_state=42)
nn.fit(X_train_s, y_train)
y_pred = nn.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {acc:.4f} ({acc*100:.1f}%)")
print(f"Iterations: {nn.n_iter_}, Loss: {nn.loss_:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Failed', 'Passed'])}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# Architecture comparison
print("\nArchitecture Comparison:")
for arch in [(8,), (16,), (8, 4), (16, 8)]:
    a = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42).fit(X_train_s, y_train)
    print(f"  {arch}: {accuracy_score(y_test, a.predict(X_test_s)):.4f}")

# Plots
plt.figure(figsize=(10,5)); plt.plot(nn.loss_curve_,'b-'); plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.title('Training Curve'); plt.savefig('/Users/sapatmohit18/Desktop/ML/neural_network_loss.png'); plt.close()
plt.figure(figsize=(8,6)); plt.imshow(cm, cmap='Blues'); plt.colorbar(); plt.xticks([0,1],['Failed','Passed']); plt.yticks([0,1],['Failed','Passed']); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'CM ({acc*100:.0f}%)'); plt.savefig('/Users/sapatmohit18/Desktop/ML/neural_network_cm.png'); plt.close()
print("\nDone!")