import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import time
import os

# Parametrii
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
win_len = 168
horizon = 48
hidden = 128
batch_size = 64
epochs = 20
lr = 0.001
l2_lambda = 1e-4
clip_value = 1.0
patience = 5

model_path = f"best_model_manual_W{win_len}_H{horizon}.pt"

print(f"Experiment: W={win_len}, H={horizon}")


def sigmoid(x):
    """ Sigmoid: 1 / (1 + e^-x) """
    return 1.0 / (1.0 + torch.exp(-x))


def tanh(x):
    """ Tanh: (e^x - e^-x) / (e^x + e^-x) """
    p = torch.exp(x)
    m = torch.exp(-x)
    return (p - m) / (p + m)


def xavier(shape):
    fan_in = shape[0]
    fan_out = shape[1]

    limit = np.sqrt(6.0 / (fan_in + fan_out))

    return (torch.rand(shape, device=device) * 2.0 * limit) - limit


def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, mae, mape, r2


class Adam:
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                grad = p.grad + self.wd * p
                self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
                self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grad ** 2)
                m_hat = self.m[i] / (1 - self.b1 ** self.t)
                v_hat = self.v[i] / (1 - self.b2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.H = hidden_size
        self.W = xavier((hidden_size, 3 * hidden_size))

        self.U = xavier((input_size, 3 * hidden_size))

        # b: Bias (initializat cu 0)
        self.b = torch.zeros(3 * hidden_size, device=device)

        # Wy: Hidden -> Output (shape: hidden, output)
        self.Wy = xavier((hidden_size, output_size))

        # by: Bias Output (initializat cu 0)
        self.by = torch.zeros(output_size, device=device)

        self.params = [self.W, self.U, self.b, self.Wy, self.by]
        for p in self.params:
            p.requires_grad_(True)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.H, device=x.device)

        # Pre-calculam input projection pentru viteza: xU + b
        xU_full = torch.matmul(x, self.U) + self.b

        for t in range(seq_len):
            # Calculam portile: h * W + input_precalculat
            gates = torch.matmul(h, self.W[:, :2 * self.H]) + xU_full[:, t, :2 * self.H]

            # --- 2. Activare Sigmoid Manuala ---
            z = sigmoid(gates[:, :self.H])  # Update gate
            r = sigmoid(gates[:, self.H:2 * self.H])  # Reset gate

            # Calculam candidatul
            cand_input = torch.matmul(h * r, self.W[:, 2 * self.H:]) + xU_full[:, t, 2 * self.H:]

            # --- 3. Activare Tanh Manuala ---
            cand = tanh(cand_input)

            h = (1 - z) * h + z * cand

        return torch.matmul(h, self.Wy) + self.by


def get_predictions_batched(model, data, batch_size=128):
    model_preds = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            pred = model.forward(batch)
            model_preds.append(pred.cpu().numpy())
    return np.concatenate(model_preds, axis=0)


def create_windows(data, wl, hz):
    n = len(data) - wl - hz + 1
    indices = np.arange(wl + hz)[None, :] + np.arange(n)[:, None]
    win = data[indices]
    return win[:, :wl], win[:, wl:wl + hz]


if __name__ == "__main__":
    # Incarcare date
    df = pd.read_csv("/kaggle/input/hourly-energy-consumption/AEP_hourly.csv")
    energy = df["AEP_MW"].values.astype(np.float32)

    # Split date
    n_total = len(energy)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_data = energy[:n_train]
    val_data = energy[n_train:n_train + n_val]
    test_data = energy[n_train + n_val:]

    # Creare ferestre
    X_raw_train, Y_raw_train = create_windows(train_data, win_len, horizon)
    X_raw_val, Y_raw_val = create_windows(val_data, win_len, horizon)
    X_raw_test, Y_raw_test = create_windows(test_data, win_len, horizon)

    # Normalizare
    train_mean, train_std = train_data.mean(), train_data.std()

    X_tr = torch.from_numpy((X_raw_train - train_mean) / train_std) \
        .reshape(-1, win_len, 1).to(device)
    Y_tr = torch.from_numpy((Y_raw_train - train_mean) / train_std).to(device)

    X_vl = torch.from_numpy((X_raw_val - train_mean) / train_std) \
        .reshape(-1, win_len, 1).to(device)
    Y_vl = torch.from_numpy((Y_raw_val - train_mean) / train_std).to(device)

    X_ts = torch.from_numpy((X_raw_test - train_mean) / train_std) \
        .reshape(-1, win_len, 1).to(device)
    Y_ts = torch.from_numpy((Y_raw_test - train_mean) / train_std).to(device)

    n_train = len(X_tr)

    # Initializare model si optimizer
    model = GRU(1, hidden, horizon)
    optimizer = Adam(model.params, lr=lr, weight_decay=l2_lambda)

    train_losses, val_losses = [], []
    best_v_loss = float("inf")
    early_stop_cnt = 0
    best_params = [p.clone() for p in model.params]

    # Training loop
    for ep in range(epochs):
        start_time = time.time()
        total_train_loss, num_batches = 0, 0
        idx = torch.randperm(len(X_tr))

        for i in range(0, len(X_tr), batch_size):
            b_idx = idx[i:i + batch_size]
            pred = model.forward(X_tr[b_idx])
            loss = torch.mean((pred - Y_tr[b_idx]) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.params, clip_value)
            optimizer.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / num_batches

        # Validare
        with torch.no_grad():
            val_loss = 0.0
            val_batches = 0
            for i in range(0, len(X_vl), batch_size):
                xb = X_vl[i:i + batch_size]
                yb = Y_vl[i:i + batch_size]
                pred = model.forward(xb)
                val_loss += torch.mean((pred - yb) ** 2).item()
                val_batches += 1
            v_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(v_loss)

        print(f"Ep {ep + 1:02d} | Train: {avg_train_loss:.5f} | Val: {v_loss:.5f} | {time.time() - start_time:.1f}s")

        # Early stopping
        if v_loss < best_v_loss:
            best_v_loss = v_loss
            early_stop_cnt = 0
            best_params = [p.clone() for p in model.params]
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                break

    # Restaurare best model
    for i, p in enumerate(model.params):
        p.data.copy_(best_params[i].data)

    # Salvare model
    checkpoint = {
        'W': model.W.data,
        'U': model.U.data,
        'b': model.b.data,
        'Wy': model.Wy.data,
        'by': model.by.data,
        'train_mean': train_mean,
        'train_std': train_std
    }
    torch.save(checkpoint, model_path)
    print("Model salvat:", model_path)

    # Generare rezultate finale
    ts_pred_raw = get_predictions_batched(model, X_ts, batch_size)
    ts_pred_gw = (ts_pred_raw * train_std + train_mean) / 1000.0
    ts_true_gw = (Y_ts.cpu().numpy() * train_std + train_mean) / 1000.0
    mse_ts, mae_ts, mape_ts, r2_ts = calculate_metrics(ts_true_gw, ts_pred_gw)

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train_mse', color='#1f77b4')
    plt.plot(val_losses, label='val_mse', color='#ff7f0e')
    plt.xlabel('Epoca')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f"loss_final_v3_W{win_len}_H{horizon}.png", dpi=300)
    plt.show()

    # Plot prediction vs real
    plt.figure(figsize=(10, 5))
    plt.plot(ts_true_gw[0], label="Real", color='#1f77b4', linewidth=2)
    plt.plot(ts_pred_gw[0], label="Predictie", color='#ff7f0e', linewidth=2)
    plt.ylabel("Consum")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(f"test_plot_final_v3_W{win_len}_H{horizon}.png", dpi=300)
    plt.show()
    print("Graficele salvate.")

    print(f"MSE:  {mse_ts:.6f}")
    print(f"MAE:  {mae_ts:.6f}")
    print(f"MAPE: {mape_ts:.2f}%")
    print(f"R2:   {r2_ts:.4f}")