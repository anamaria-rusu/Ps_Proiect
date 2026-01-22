import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1) Date + split
# -------------------------
cale_fisier = "data/data.csv"  # schimba

df = pd.read_csv(cale_fisier, sep=";")
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["datetime"]).sort_values("datetime")

df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
df = df.dropna(subset=["Global_active_power"]).reset_index(drop=True)

n = len(df)
n_train = int(0.8 * n)
train_df = df.iloc[:n_train].copy()
val_df   = df.iloc[n_train:].copy()

# normalizare (fit pe train)
mu = train_df["Global_active_power"].mean()
std = train_df["Global_active_power"].std() + 1e-6
train_df["P_norm"] = (train_df["Global_active_power"] - mu) / std
val_df["P_norm"]   = (val_df["Global_active_power"] - mu) / std

# -------------------------
# 2) Ferestre
# -------------------------
def ferestre(df_part, T=60):
    s = df_part["P_norm"].values.astype(np.float32)
    t = df_part["datetime"].values
    X, y, y_time = [], [], []
    for i in range(T, len(s)):
        X.append(s[i-T:i])
        y.append(s[i])
        y_time.append(t[i])
    X = np.array(X, dtype=np.float32)[:, :, None]  # (N, T, 1)
    y = np.array(y, dtype=np.float32)[:, None]     # (N, 1)
    y_time = pd.to_datetime(y_time)
    return X, y, y_time

T = 60
X_train, y_train, _ = ferestre(train_df, T=T)
X_val, y_val, val_time = ferestre(val_df, T=T)

# -------------------------
# 3) Dataset + DataLoader
# -------------------------
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # float32
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 256  # mare = mai rapid pe GPU (de obicei)
train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(WindowDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

# -------------------------
# 4) Model LSTM
# -------------------------
class LSTMReg(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)      # out: (batch, T, hidden)
        h_last = out[:, -1, :]     # ultimul pas
        y = self.fc(h_last)        # (batch, 1)
        return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMReg(hidden=32).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# 5) Train loop (minimal)
# -------------------------
def train_epoch():
    model.train()
    total = 0.0
    for Xb, yb in train_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(Xb)
    return total / len(train_loader.dataset)

@torch.no_grad()
def eval_epoch():
    model.eval()
    total = 0.0
    for Xb, yb in val_loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        pred = model(Xb)
        loss = criterion(pred, yb)
        total += loss.item() * len(Xb)
    return total / len(val_loader.dataset)

epochs = 10
for ep in range(1, epochs+1):
    tr = train_epoch()
    va = eval_epoch()
    print(f"Epoch {ep:02d} | train MSE: {tr:.5f} | val MSE: {va:.5f}")

# -------------------------
# 6) Predict pe val (pentru grafic)
# -------------------------
@torch.no_grad()
def predict_all(X):
    model.eval()
    preds = []
    loader = DataLoader(WindowDataset(X, np.zeros((len(X),1), np.float32)),
                        batch_size=batch_size, shuffle=False)
    for Xb, _ in loader:
        Xb = Xb.to(device)
        preds.append(model(Xb).cpu().numpy())
    return np.vstack(preds)

y_pred_val = predict_all(X_val)

# denormalizare la kW
y_pred_kw = y_pred_val * std + mu

train_series = pd.Series(train_df["Global_active_power"].values, index=train_df["datetime"])
val_series   = pd.Series(val_df["Global_active_power"].values,   index=val_df["datetime"])
pred_series  = pd.Series(y_pred_kw.flatten(), index=val_time)

plt.figure(figsize=(12,5))
plt.plot(train_series.index, train_series.values, label="Train")
plt.plot(val_series.index, val_series.values, label="Val")
plt.plot(pred_series.index, pred_series.values, label="Predictions")
plt.title("Model LSTM (PyTorch)")
plt.xlabel("Date")
plt.ylabel("Global_active_power (kW)")
plt.legend()
plt.tight_layout()
plt.show()
