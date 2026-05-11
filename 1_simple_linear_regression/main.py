"""
Simple Linear Regression — Generalized
Auto-detects: first numeric column as X, last numeric column as Y.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "data.csv"   # <-- change this path to your dataset


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Columns: {list(df.columns)}")

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number]).dropna()

    if df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns (one feature, one target).")

    # First numeric col = X, last numeric col = Y
    x_col = df.columns[0]
    y_col = df.columns[-1]
    X = df[[x_col]].values
    y = df[y_col].values

    print(f"\nFeature (X): {x_col}")
    print(f"Target  (Y): {y_col}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nEquation: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE:      {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, color='#2196F3', alpha=0.6, label='Actual')
    plt.plot(X_test, y_pred, color='#EF5350', linewidth=2, label='Predicted')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
