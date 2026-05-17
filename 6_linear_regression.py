import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dataset
df = pd.DataFrame({'Area': [1000,1500,1200,1800,2000,900,1600,1400,1700,1100,2200,1300,1900,1050,1750],
                   'Bedrooms': [2,3,2,4,4,1,3,3,4,2,5,2,4,2,4],
                   'Age': [10,5,8,3,2,15,6,7,4,12,1,9,3,11,5],
                   'Distance': [5,8,6,10,12,3,7,9,11,4,14,5,10,4,8],
                   'Price': [150,220,170,280,320,120,230,200,270,140,350,160,290,135,260]})
print("Data:\n", df)

print("\nCorrelations with Price:")
print(df.corr()['Price'].sort_values(ascending=False))

X, y = df.drop('Price',axis=1), df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

model = LinearRegression().fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(f"\nMSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")

print("\nCoefficients:")
for feat, coef in zip(X.columns, model.coef_): print(f"  {feat}: {coef:.3f}")
print(f"  Intercept: {model.intercept_:.3f}")

# Plots
fig, axes = plt.subplots(2,2, figsize=(12,10))
axes[0,0].scatter(y_test, y_pred, s=100); axes[0,0].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--'); axes[0,0].set_xlabel('Actual'); axes[0,0].set_ylabel('Predicted'); axes[0,0].set_title('Actual vs Predicted')
axes[0,1].scatter(y_pred, y_test-y_pred, s=100); axes[0,1].axhline(y=0, color='r', linestyle='--'); axes[0,1].set_xlabel('Predicted'); axes[0,1].set_ylabel('Residual'); axes[0,1].set_title('Residual Plot')
axes[1,0].barh(X.columns, np.abs(model.coef_)); axes[1,0].set_xlabel('|Coefficient|'); axes[1,0].set_title('Feature Importance')
axes[1,1].hist(y_test-y_pred, bins=8, edgecolor='black'); axes[1,1].set_xlabel('Residual'); axes[1,1].set_title('Residual Distribution')
plt.tight_layout(); plt.savefig('linear_regression_analysis.png'); plt.close()
print("\nDone!")