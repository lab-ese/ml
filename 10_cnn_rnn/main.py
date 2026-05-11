"""
CNN + RNN — Generalized (PyTorch)

CNN — auto-detects dataset type and adapts architecture:
  • Folder path → 2D CNN on images (one subfolder per class)
  • CSV with pixel-square columns → 2D CNN on images (label + flattened pixels)
  • CSV with text column → 1D CNN on character-level tokens (text classification)
  • CSV with numeric features → 1D CNN on feature vector

RNN — single-column time series CSV — predicts next value from past SEQ_LEN values.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# CNN_PATH can be: folder of images / image-pixel CSV / text CSV / numeric-features CSV
# Convention for CSV: first column = label, remaining columns = data
CNN_PATH = "data.csv"             # <-- change for CNN dataset
RNN_CSV_PATH = "series.csv"       # <-- change for time series (single numeric column)

IMG_SIZE = 28                      # used when loading from a folder of images
SEQ_LEN = 12                       # window size for RNN


# ============================================================
# CNN MODELS
# ============================================================

class CNN2D(nn.Module):
    """2D CNN for images."""

    def __init__(self, img_size, n_classes, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = 32 * (img_size // 4) * (img_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN1D(nn.Module):
    """1D CNN for sequence/feature data (numeric or token embeddings)."""

    def __init__(self, seq_len, n_classes, in_channels=1, embed_size=None, vocab_size=None):
        super().__init__()
        self.embed = None
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, embed_size)
            in_channels = embed_size

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        if self.embed is not None:
            x = self.embed(x).transpose(1, 2)   # (B, L) -> (B, L, E) -> (B, E, L)
        return self.classifier(self.features(x))


# ============================================================
# CNN — DATA LOADERS
# ============================================================

def load_image_folder(path):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(path, transform=transform)
    print(f"Loaded image folder: {len(ds)} images, {len(ds.classes)} classes ({ds.classes})")
    X = np.stack([img.numpy() for img, _ in ds]).astype(np.float32)
    y = np.array([lbl for _, lbl in ds], dtype=np.int64)
    return ('image', X, y, IMG_SIZE)


def load_csv_dataset(path):
    df = pd.read_csv(path).dropna()
    print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} cols")

    # First column = label
    y_raw = df.iloc[:, 0].values
    rest = df.iloc[:, 1:]

    # Encode labels
    if y_raw.dtype == object:
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
    else:
        y = y_raw.astype(np.int64)

    # Detect text columns
    text_cols = rest.select_dtypes(include=[object]).columns.tolist()

    if text_cols:
        # ---- Text mode: char-level CNN ----
        print(f"Mode: TEXT — using columns {text_cols}")
        texts = rest[text_cols].astype(str).agg(' '.join, axis=1).tolist()
        max_len = min(200, max(len(t) for t in texts))
        # char vocab
        chars = sorted(set(''.join(texts)))
        char2idx = {c: i + 1 for i, c in enumerate(chars)}   # 0 = padding
        X = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:max_len]):
                X[i, j] = char2idx.get(c, 0)
        return ('text', X, y, len(char2idx) + 1)

    # ---- Numeric mode ----
    X_num = rest.select_dtypes(include=[np.number]).values.astype(np.float32)
    if X_num.shape[1] == 0:
        raise ValueError("No usable feature columns found.")

    # Perfect square pixel count → image
    img_size = int(np.sqrt(X_num.shape[1]))
    if img_size * img_size == X_num.shape[1] and img_size >= 8:
        print(f"Mode: IMAGE — {img_size}×{img_size} pixels")
        X_max = X_num.max() if X_num.max() > 1 else 1
        X = (X_num / X_max).reshape(-1, 1, img_size, img_size)
        return ('image', X, y, img_size)

    # Else → 1D numeric features
    print(f"Mode: NUMERIC FEATURES — {X_num.shape[1]} features (1D CNN)")
    # Normalize
    mean, std = X_num.mean(0), X_num.std(0) + 1e-9
    X = ((X_num - mean) / std).reshape(-1, 1, X_num.shape[1])
    return ('numeric', X, y, X_num.shape[1])


# ============================================================
# CNN — TRAINING
# ============================================================

def train_cnn(epochs=5, batch_size=64):
    print("=" * 50)
    print("  CNN")
    print("=" * 50)

    if os.path.isdir(CNN_PATH):
        mode, X, y, meta = load_image_folder(CNN_PATH)
    elif os.path.isfile(CNN_PATH):
        mode, X, y, meta = load_csv_dataset(CNN_PATH)
    else:
        raise FileNotFoundError(f"CNN_PATH '{CNN_PATH}' not found.")

    n_classes = len(np.unique(y))
    print(f"Samples: {len(X)}  |  Classes: {n_classes}")

    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    Xtr, Xte = X[idx[:split]], X[idx[split:]]
    ytr, yte = y[idx[:split]], y[idx[split:]]

    if mode == 'image':
        model = CNN2D(img_size=meta, n_classes=n_classes, in_channels=Xtr.shape[1])
        Xtr_t, Xte_t = torch.tensor(Xtr), torch.tensor(Xte)
    elif mode == 'text':
        model = CNN1D(seq_len=Xtr.shape[1], n_classes=n_classes,
                      embed_size=32, vocab_size=meta)
        Xtr_t, Xte_t = torch.tensor(Xtr, dtype=torch.long), torch.tensor(Xte, dtype=torch.long)
    else:  # numeric
        model = CNN1D(seq_len=Xtr.shape[2], n_classes=n_classes, in_channels=1)
        Xtr_t, Xte_t = torch.tensor(Xtr), torch.tensor(Xte)

    train_loader = DataLoader(TensorDataset(Xtr_t, torch.tensor(ytr)),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte_t, torch.tensor(yte)),
                             batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_accs = [], []
    print(f"\n  {'Epoch':<8} {'Loss':<12} {'Test Acc'}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total * 100
        train_losses.append(total_loss / len(train_loader))
        test_accs.append(acc)
        print(f"  {epoch+1:<8} {train_losses[-1]:<12.4f} {acc:.2f}%")

    return train_losses, test_accs


# ============================================================
# RNN — Generalized
# ============================================================

class RNN(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(1, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def train_rnn(epochs=100, seq_len=SEQ_LEN):
    print("\n" + "=" * 50)
    print("  RNN")
    print("=" * 50)

    df = pd.read_csv(RNN_CSV_PATH)
    series = df.select_dtypes(include=[np.number]).iloc[:, 0].values.astype(np.float32)
    print(f"Loaded: {len(series)} time steps")

    s_min, s_max = series.min(), series.max()
    data = (series - s_min) / (s_max - s_min + 1e-9)

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = int(len(X) * 0.8)
    Xtr = torch.tensor(X[:split]).unsqueeze(-1)
    Xte = torch.tensor(X[split:]).unsqueeze(-1)
    ytr = torch.tensor(y[:split]).unsqueeze(-1)
    yte = torch.tensor(y[split:]).unsqueeze(-1)

    print(f"Train sequences: {len(Xtr)}  |  Test sequences: {len(Xte)}")

    model = RNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses, test_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(Xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            tloss = criterion(model(Xte), yte).item()
        train_losses.append(loss.item())
        test_losses.append(tloss)
        if epoch in (1, 25, 50, 75, 100):
            print(f"  Epoch {epoch:>3}  train MSE: {loss.item():.6f}  test MSE: {tloss:.6f}")

    model.eval()
    with torch.no_grad():
        yp = model(Xte).numpy().ravel()
    yp_real = yp * (s_max - s_min) + s_min
    yt_real = y[split:] * (s_max - s_min) + s_min
    rmse = np.sqrt(np.mean((yt_real - yp_real) ** 2))
    print(f"\nTest RMSE: {rmse:.4f}")
    return train_losses, test_losses, yt_real, yp_real


def plot_all(cnn_losses, cnn_accs, rnn_train_losses, rnn_test_losses, yt, yp):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(cnn_losses, color='#EF5350', linewidth=2, marker='o')
    axes[0, 0].set_title('CNN Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(cnn_accs, color='#4CAF50', linewidth=2, marker='s')
    axes[0, 1].set_title('CNN Test Accuracy (%)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(rnn_train_losses, color='#2196F3', linewidth=2, label='Train')
    axes[1, 0].plot(rnn_test_losses, color='#EF5350', linewidth=2, linestyle='--', label='Test')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('RNN Loss Curves')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(yt, color='#2196F3', linewidth=2, label='Actual')
    axes[1, 1].plot(yp, color='#EF5350', linewidth=2, linestyle='--', label='Predicted')
    axes[1, 1].set_title('RNN Predictions on Test Set')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('output.png', dpi=120)
    print("\nPlot saved: output.png")


if __name__ == '__main__':
    cnn_losses, cnn_accs = train_cnn(epochs=5)
    rnn_train_losses, rnn_test_losses, yt, yp = train_rnn(epochs=100)
    plot_all(cnn_losses, cnn_accs, rnn_train_losses, rnn_test_losses, yt, yp)
