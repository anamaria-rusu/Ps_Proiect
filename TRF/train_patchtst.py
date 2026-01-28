import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchtst import PatchTST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# se creaza samples astfel incat
# inputul este o fereastra de dim win_len
# label-ul este o fereastra de dim horizon
# de exemplu pentru secventa [t0, t1, t2, t3, t4, t5, t6] o impartire poaet fi
# x = [t0, t1, t2, t3, t4], y = [t5, t6] pentru win_len=3 si horizon=2
def create_samples(data, win_len, horizon):
    x, y = [], []
    for i in range(len(data) - win_len - horizon + 1):
        x.append(data[i : i+win_len])
        y.append(data[i+win_len : i+win_len+horizon])
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)



def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def mape(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# hiperparametri
win_len = 336 #lungimea ferestrei de intrare
horizon = 24 #lungimea predictiei
in_channels = 1 #nr features
patch_len = 16       
stride = 8          
d_model = 256        
num_heads = 4 #nr attention heads
num_layers = 3 #nr encoder layers
d_ff = 1024 #dimensiune FFN
dropout = 0.1 

lr = 0.001           
epochs = 50        
batch_size = 32     
patience = 5 #early stopping patience
delta = 1e-4 #pt early stopping


#incarcarea datelor
print("-----Incarcare date")
data = pd.read_csv("data/AEP_hourly.csv")

data["time"] = pd.to_datetime(data["time"])
data = data.set_index("time")
energy = data["energy"]
print(f"Date: {len(energy)} puncte")


#split
n = len(energy)
train_data = energy.values[:int(n*0.7)]
val_data = energy.values[int(n*0.7):int(n*0.9)]
test_data = energy.values[int(n*0.9):]


x_train, y_train = create_samples(train_data, win_len, horizon)
x_val, y_val = create_samples(val_data, win_len, horizon)
x_test, y_test = create_samples(test_data, win_len, horizon)

x_train /= 1000.0; y_train /= 1000.0
x_val /= 1000.0; y_val /= 1000.0
x_test /= 1000.0; y_test /= 1000.0


print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


#reshape pentru a adauga dimensiunea channel
#din (N, win_len) la (N, win_len, 1)
x_train = x_train.reshape((-1, win_len, in_channels))
x_val = x_val.reshape((-1, win_len, in_channels))
x_test = x_test.reshape((-1, win_len, in_channels))
y_train = y_train.reshape((-1, horizon, in_channels))
y_val = y_val.reshape((-1, horizon, in_channels))
y_test = y_test.reshape((-1, horizon, in_channels))


#model PatchTST
print("\n------Initializare model")
model = PatchTST(
    seq_len=win_len,
    pred_len=horizon,
    in_channels=in_channels,
    patch_len=patch_len,
    stride=stride,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    dropout=dropout
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parametri: {total_params:,}")


#loss si optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


#early stopping + checkpoint model
early_count = 0
early_val = float('inf')
best_model_state = None
best_epoch = 0

#pentru salvare metrici
train_losses = []
val_losses = []
val_maes = []


# training
print("\n------Antrenare")

#directorul pentru rezultate
os.makedirs("TRF/results", exist_ok=True)

start_time = time.time()

for e in range(epochs):
    epoch_start = time.time()
    losses = []

    #shuffle
    p = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[p]
    y_train_shuffled = y_train[p]

    #procesare pe batch-uri cu progress bar
    num_batches = len(x_train) - batch_size + 1
    pbar = tqdm(range(0, num_batches, batch_size), 
                desc=f"Epoch {e+1}/{epochs}", 
                unit="batch",
                ncols=80)
    
    for i in pbar:
        x_batch = torch.from_numpy(x_train_shuffled[i : i+batch_size]).to(device)
        y_batch = torch.from_numpy(y_train_shuffled[i : i+batch_size]).to(device)

        #forward
        y_hat = model(x_batch)

        #loss (mse)
        loss = criterion(y_batch, y_hat)

        #backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #l2 gradient clipping
        optimizer.step()

        losses.append(float(loss.item()))
        
        #update progress bar cu loss curent
        pbar.set_postfix({'loss': f'{np.mean(losses):.6f}'})
    
    mse_train = float(np.mean(losses))
    train_losses.append(mse_train)

    #validare
    with torch.no_grad():
        x_batch = torch.from_numpy(x_val).to(device)
        y_batch = torch.from_numpy(y_val).to(device)
        y_hat = model(x_batch)
        mse_val = float(criterion(y_batch, y_hat).item())
        mae_val = mae(y_batch.cpu().numpy(), y_hat.cpu().numpy())
    
    val_losses.append(mse_val)
    val_maes.append(mae_val)
    
    #clear CUDA cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    #calcula timp estimat pentru rest
    avg_time_per_epoch = total_time / (e + 1)
    remaining_epochs = epochs - (e + 1)
    eta_time = avg_time_per_epoch * remaining_epochs

    print(f"\n[Epoch {e+1}/{epochs}] Timp: {epoch_time:.2f}s")
    print(f"Train MSE: {mse_train:.6f}")
    print(f"Val MSE:   {mse_val:.6f}")
    print(f"Val MAE:   {mae_val:.6f}")
    print(f"Total: {total_time/60:.1f}m | Timp ramas: {eta_time/60:.1f}m")

    #early stopping + checkpoint
    if mse_val < early_val - delta:
        early_val = mse_val
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        best_epoch = e + 1
        torch.save(best_model_state, "TRF/results/patchtst.pth")
        early_count = 0
        print(f"Best model salvat (val_loss: {mse_val:.6f})")
    else:
        early_count += 1
        if early_count >= patience:
            print(f"\nEarly stopping la epoch {e+1}")
            break


#restaurare best model
saved_state = torch.load("TRF/results/patchtst.pth")
model.load_state_dict(saved_state)
print(f"Best model: epoch {best_epoch}")


#testare
print("\n-----Test")

with torch.no_grad():
    x_batch = torch.from_numpy(x_test.astype(np.float32)).to(device)
    y_hat_all = model(x_batch)

# converteste la numpy pentru metrici
y_hat_all = y_hat_all.detach().cpu().numpy()
y_true_all = y_test

# metrici pe tot setul de test
mse_test = mse(y_true_all, y_hat_all)
mae_test = mae(y_true_all, y_hat_all)
r2_test = r2_score(y_true_all, y_hat_all)
mape_test = mape(y_true_all, y_hat_all)

print(f"Test MSE:  {mse_test:.6f}")
print(f"Test MAE:  {mae_test:.6f}")
print(f"Test R2:   {r2_test:.6f}")
print(f"Test MAPE: {mape_test:.2f}%")

# primul sample pentru vizualizare
y_true = y_true_all[0].flatten()
y_hat = y_hat_all[0].flatten()


#training history
plt.figure(figsize=(12, 4))
plt.plot(train_losses, label="Train MSE", marker='o')
plt.plot(val_losses, label="Val MSE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("TRF/results/training_history.pdf", format='pdf')
plt.close()



#predictions vs true
plt.figure(figsize=(12,4))
plt.plot(y_true, label="true")
plt.plot(y_hat, label="pred")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("TRF/results/test_prediction.pdf", format='pdf')
plt.close()
print("\nRezultate salvate in TRF/results/")
