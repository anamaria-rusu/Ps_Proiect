import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lstm manual (in afara de pasul de backpropagation)
class lstm:

    # calcul sigmoida
    def sigmoid(self, x):
        return torch.sigmoid(x)
    

    # calcul tanh
    def tanh(self, x):
        return torch.tanh(x)
    

    # initializare greutati cu Xavier
    def xavier(self, fan_in, fan_out):
        lim = (6.0 / (fan_in + fan_out)) ** 0.5
        w = np.random.uniform(-lim, lim, (fan_in, fan_out)).astype(np.float32)
        return torch.tensor(w, device = device, requires_grad=True)
    

    # parametrii retelei
    def __init__(self, input, hidden, output):
        self.input = input
        self.hidden = hidden
        self.output = output

        # greutatile
        self.Wf = self.xavier(input + hidden, hidden)
        self.Wi = self.xavier(input + hidden, hidden)
        self.Wo = self.xavier(input + hidden, hidden)
        self.Wc = self.xavier(input + hidden,hidden)
        self.Wy = self.xavier(hidden, output)

        # bias
        self.bf = torch.ones(hidden, device=device, requires_grad=True)
        self.bi = torch.zeros(hidden, device=device, requires_grad=True)
        self.bc = torch.zeros(hidden, device=device, requires_grad=True)
        self.bo = torch.zeros(hidden, device=device, requires_grad=True)
        self.by = torch.zeros(output, device=device, requires_grad=True)


    # forward
    def forward(self, batch):
        # dim batch-ului (cate inputuri sunt in batch si dim fereastrei)
        batch_size = batch.shape[0]
        win_size = batch.shape[1]

        # reinitializare h si c pentru fiecare batch
        h = torch.zeros(batch_size, self.hidden, device=device)
        c = torch.zeros(batch_size, self.hidden, device=device)

        # parcurgere fereastra
        for i in range(win_size):
            x_t = batch[:, i, :]                
            z = torch.cat((h, x_t), dim=1)                
            f_t = self.sigmoid(z @ self.Wf + self.bf)     
            i_t = self.sigmoid(z @ self.Wi + self.bi)    
            o_t = self.sigmoid(z @ self.Wo + self.bo)    
            g_t = self.tanh(z @ self.Wc + self.bc)        
            c = f_t * c + i_t * g_t                       
            h = o_t * self.tanh(c)                        
        y = h @ self.Wy + self.by                      
        return y



def scale_energy_simple(energy, factor=1000.0):
    return (energy / factor).astype(np.float32)


# se creaza sampels astfel incat
# inputul este o feresatra de dim win_len
# label-ul este o fereastra de dim horizon
# de exemplu pentru secventa [t0, t1, t2, t3, t4, t5, t6] o impartire poaet fi
# x = [t0, t1, t2, t3, t4], y = [t5, t6] pentru win_len=3 si horizon=2
def create_samples(data, win_len, horizon):
    x, y = [], []
    for i in range(len(data) - win_len - horizon + 1):
        x.append(data[i : i+win_len])
        y.append(data[i+win_len : i+win_len+horizon])
    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)



# gradient clipping pt a preveni gradient exploding 
def gradient_clipping(params, threshold=1.0, eps=1e-6):
    l2_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        l2_sq += float(torch.sum(p.grad * p.grad).item())

    l2 = l2_sq ** 0.5
    f = threshold / (l2 + eps)

    if f < 1.0:
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(f)



# hiperparametri
win_len = 168
horizon = 48
hidden = 128
lr = 0.001
epochs = 20
batch_size = 64
patience = 5
delta = 1e-4


# incarcarea datelor
data = pd.read_csv("/kaggle/input/energyset/AEP_hourly.csv")
data["time"] = pd.to_datetime(data["time"])
data = data.set_index("time")
energy = data["energy_kWh"].values
energy = scale_energy_simple(energy)


# split date 
n = len(energy)
energy_train = energy[:int(0.8 * n)]
energy_val   = energy[int(0.8 * n):int(0.9 * n)]
energy_test  = energy[int(0.9 * n):]


# create_samples 
x_train, y_train = create_samples(energy_train, win_len, horizon)
x_val,   y_val   = create_samples(energy_val,   win_len, horizon)
x_test,  y_test  = create_samples(energy_test,  win_len, horizon)


# normalizare (doar din train)
mean = x_train.mean()
std = x_train.std() + 1e-8

x_train = (x_train - mean) / std
x_val   = (x_val   - mean) / std
x_test  = (x_test  - mean) / std

y_train = (y_train - mean) / std
y_val   = (y_val   - mean) / std
y_test  = (y_test  - mean) / std


# reshape
x_train = x_train.reshape((-1, win_len, 1))
x_val   = x_val.reshape((-1, win_len, 1))
x_test  = x_test.reshape((-1, win_len, 1))


# model lstm
model = lstm(1, hidden, horizon)
early_count = 0
early_val = float('inf')
early_params = None
model_params = [model.Wf, model.Wc, model.Wi, model.Wo, model.Wy,model.bf, model.bc, model.bi, model.bo, model.by]

# adam
m = [torch.zeros_like(p) for p in model_params]  
v = [torch.zeros_like(p) for p in model_params]  
beta1, beta2 = 0.9, 0.999
t = 0


# istoric mse train+val
train_history = []
val_history = []


# train
for e in range(epochs):
    losses = []

    # shuffle la ferestre (ORDINEA DIN INT FERESTREI NU SE MODIFICA)
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]    
    y_train = y_train[p]

    # procesam datele pe batch-uri
    for i in range(0, len(x_train), batch_size):

        # luam cate un bacth de dim batch_size
        x_batch = torch.from_numpy(x_train[i : i+batch_size]).to(device)
        y_batch = torch.from_numpy(y_train[i : i+batch_size]).to(device)

        # obtinem predictiile
        y_hat = model.forward(x_batch)

        # calculcam loss-ul (mse)
        loss = torch.mean((y_batch - y_hat)**2)

        # backprop din librarii (autograd)
        loss.backward()

        # gradient clipping
        gradient_clipping(model_params)

        # optimizator adam
        t += 1
        with torch.no_grad():
            for j, p_ in enumerate(model_params):
                g = p_.grad  # gradient curent
        
                # momente
                m[j] = beta1 * m[j] + (1.0 - beta1) * g
                v[j] = beta2 * v[j] + (1.0 - beta2) * (g * g)
        
                # bias
                m_hat = m[j] / (1.0 - beta1 ** t)
                v_hat = v[j] / (1.0 - beta2 ** t)
        
                # update
                p_ -= lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
        
                # reset
                p_.grad.zero_()

        losses.append(float(loss.item()))
    mse_train = float(np.mean(losses))


    # validare
    with torch.no_grad():
        x_batch = torch.from_numpy(x_val).to(device)
        y_batch = torch.from_numpy(y_val).to(device)
        y_hat = model.forward(x_batch)
        mse_val = float(torch.mean((y_batch - y_hat) ** 2).item())

    train_history.append(mse_train)
    val_history.append(mse_val)
    print(f"e {e+1} mse_train {mse_train} mse_val {mse_val}")


    # early stopping + checkpoint model
    if mse_val < early_val - delta:
        early_val = mse_val
        with torch.no_grad():
            early_params = [p.clone() for p in model_params]
        torch.save(early_params, "/kaggle/working/lstm_48_168.pth")
        early_count = 0
    else:
        early_count += 1
        if early_count >= patience:
            break


# salvare csv (epoca, train_mse, val_mse)
df_hist = pd.DataFrame({
    "epoca": np.arange(1, len(train_history) + 1),
    "train_mse": train_history,
    "val_mse": val_history
})
df_hist.to_csv("/kaggle/working/mse_history_48_168.csv", index=False)


# restaurare
saved_params = torch.load("/kaggle/working/lstm_48_168.pth")
with torch.no_grad():
    for p, bp in zip(model_params, saved_params):
        p.copy_(bp)


# testare + calcul metrici
sum_sq_error = 0.0       
sum_abs_error = 0.0        
sum_abs_pct_error = 0.0   
sum_y_true = 0.0            
sum_y_true_sq = 0.0        
num_values = 0             

with torch.no_grad():
    for i in range(0, len(x_test), batch_size):
        # input + predictii
        x_batch = torch.from_numpy(x_test[i:i+batch_size]).to(device)
        y_batch = torch.from_numpy(y_test[i:i+batch_size]).to(device)
        y_pred = model.forward(x_batch)

        # denormalizare
        y_true = y_batch * std + mean
        y_pred = y_pred * std + mean

        # metrici
        diff = (y_true - y_pred)
        sum_sq_error += float(torch.sum(diff * diff).item())
        sum_abs_error += float(torch.sum(torch.abs(diff)).item())
        sum_abs_pct_error += float(torch.sum(torch.abs(diff) / (torch.abs(y_true) + 1e-8)).item())
        sum_y_true += float(torch.sum(y_true).item())
        sum_y_true_sq += float(torch.sum(y_true * y_true).item())
        num_values += int(y_true.numel())


# metrici globale
mse = sum_sq_error / max(1, num_values)
mae = sum_abs_error / max(1, num_values)
mape = 100.0 * (sum_abs_pct_error / max(1, num_values))
mean_y = sum_y_true / max(1, num_values)
ss_tot = sum_y_true_sq - max(1, num_values) * (mean_y ** 2)
r_square = 1.0 - (sum_sq_error / ss_tot) if ss_tot > 0 else float("nan")
print(f"test:  MSE={mse}  MAE={mae}  MAPE={mape}  Rsquare={r_square}")


# grafic test[0]
start_win = x_test[0]
with torch.no_grad():
    x_batch = torch.from_numpy(start_win.reshape(1, win_len, 1)).to(device)
    y_hat = model.forward(x_batch)

# denormalizare
y_hat = y_hat.detach().cpu().numpy().reshape(-1)
y_true = y_test[0].reshape(-1) 
y_hat = y_hat * std + mean
y_true = y_true * std + mean


# grafice:
# curba loss
plt.figure(figsize=(10,4))
plt.plot(train_history, label="train_mse")
plt.plot(val_history, label="val_mse")
plt.xlabel("epoca")
plt.ylabel("mse")
plt.legend()
plt.grid(True)
plt.savefig("/kaggle/working/loss_48_168.png", dpi=150, bbox_inches="tight")
plt.show()

# predictii test[0]
plt.figure(figsize=(12,4))
plt.plot(y_true, label="true")
plt.plot(y_hat, label="pred")
plt.legend(); 
plt.grid(True)
plt.savefig("/kaggle/working/test0_48_168.png", dpi=150, bbox_inches="tight")
plt.show()

