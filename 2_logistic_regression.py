import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve

# Dataset
df = pd.DataFrame({'StudyHours': [1,2,3,4,5,6,7,8,9,10,2.5,3.5,6.5,7.5,4.5],
                   'Attendance': [60,65,70,75,80,85,90,95,98,100,68,72,88,92,78],
                   'Passed': [0,0,0,0,0,1,1,1,1,1,0,0,1,1,0]})
print("Data:\n", df)

X, y = df[['StudyHours','Attendance']], df['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
y_pred, y_prob = model.predict(X_test_s), model.predict_proba(X_test_s)[:,1]

# Metrics
print(f"\nAccuracy: {accuracy_score(y_test,y_pred):.4f}")
print(f"Precision: {precision_score(y_test,y_pred):.4f}")
print(f"Recall: {recall_score(y_test,y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test,y_prob):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test,y_pred)}")

# Plots
plt.figure(figsize=(8,6)); plt.plot(*roc_curve(y_test,y_prob)[:2], 'b-', label=f'AUC={roc_auc_score(y_test,y_prob):.4f}'); plt.plot([0,1],[0,1],'r--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.savefig('/Users/sapatmohit18/Desktop/ML/logistic_regression_roc.png'); plt.close()
plt.figure(figsize=(6,5)); plt.imshow(confusion_matrix(y_test,y_pred), cmap='Blues'); plt.colorbar(); plt.xlabel('Pred'); plt.ylabel('Actual'); plt.xticks([0,1],['Fail','Pass']); plt.yticks([0,1],['Fail','Pass']); plt.savefig('/Users/sapatmohit18/Desktop/ML/logistic_regression_cm.png'); plt.close()
print("\nDone!")