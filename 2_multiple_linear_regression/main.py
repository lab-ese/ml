"""
Multiple Linear Regression — Generalized
Auto-detects: all numeric columns except last as features, last numeric column as target.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

CSV_PATH = "data.csv"   # <-- change this path to your dataset


def encode_features(df_features):
    """Auto-handle numeric / categorical / free-text columns."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    parts, names = [], []
    for col in df_features.columns:
        s = df_features[col]
        if s.dtype == object:
            avg_len = s.astype(str).str.len().mean()
            if avg_len > 20 or s.nunique() > 50:
                vec = TfidfVectorizer(max_features=200, stop_words='english')
                m = vec.fit_transform(s.astype(str)).toarray()
                parts.append(m)
                names.extend([f'{col}_tfidf_{w}' for w in vec.get_feature_names_out()])
            else:
                d = pd.get_dummies(s, drop_first=True)
                parts.append(d.values.astype(float))
                names.extend([f'{col}={c}' for c in d.columns])
        else:
            parts.append(s.values.reshape(-1, 1).astype(float))
            names.append(col)
    return np.hstack(parts).astype(np.float32), names


def main():
    df = pd.read_csv(CSV_PATH).dropna()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    # Last column must be numeric (target for regression)
    y_col = df.columns[-1]
    if df[y_col].dtype == object:
        raise ValueError(f"Target '{y_col}' must be numeric for regression.")

    y = df[y_col].values.astype(float)
    X, feature_names = encode_features(df.drop(columns=[y_col]))

    print(f"\nFeatures: {len(feature_names)} ({feature_names[:8]}{'...' if len(feature_names) > 8 else ''})")
    print(f"Target:   {y_col}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    print(f"\nIntercept: {model.intercept_:.4f}")
    print("Top coefficients (|magnitude|):")
    coef_pairs = sorted(zip(feature_names, model.coef_), key=lambda p: abs(p[1]), reverse=True)
    for name, coef in coef_pairs[:10]:
        print(f"  {name[:40]:40s} {coef:.4f}")

    print(f"\nR² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE:      {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # Plot actual vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='#2196F3', alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, color='#EF5350', linewidth=2, linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Multiple Linear Regression — Actual vs Predicted')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    main()
