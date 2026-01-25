import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# lstm manual (in afara de oasul de backpropagation)
class lstm:

    # calcul sigmoida
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    

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
        self.bf = torch.zeros(hidden, device=device, requires_grad=True)
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



# hiperparametri
win_len = 192
horizon = 96
hidden = 64
lr = 0.001
epochs = 60
batch_size = 64
patience = 5
delta = 1e-8



# incarcarea datelor
data = pd.read_csv("data/AEP_hourly.csv")
data["time"] = pd.to_datetime(data["time"])
data = data.set_index("time")
energy = data["energy"]



# split date
x, y = create_samples(energy.values, win_len, horizon)

# total train - 90% din total
x_train_all = x[:int(0.9*len(x))] 
y_train_all = y[:int(0.9*len(y))]

# rezervat train - 80% din train
x_train = x_train_all[:int(0.8*len(x_train_all))]
y_train = y_train_all[:int(0.8*len(y_train_all))]

# rezervat validare - 20% din train
x_val = x_train_all[int(0.8*len(x_train_all)):]
y_val = y_train_all[int(0.8*len(y_train_all)):]

# total test - 10% din total
x_test = x[int(0.9*len(x)):]
y_test = y[int(0.9*len(y)):]



# normalizare cu minmax + o valoarea mica, delta, in caz ca apare / 0
min_ = x_train.min()
max_ = x_train.max()

x_train = (x_train - min_) / (max_ - min_ + delta)
x_val = (x_val - min_) / (max_ - min_ + delta)
x_test = (x_test - min_) / (max_ - min_ + delta)
y_train = (y_train - min_) / (max_ - min_ + delta)
y_val = (y_val - min_) / (max_ - min_ + delta)
y_test = (y_test - min_) / (max_ - min_ + delta)

# reshape
x_train = x_train.reshape((-1, win_len, 1))
x_val = x_val.reshape((-1, win_len, 1))
x_test = x_test.reshape((-1, win_len, 1))



# model lstm
model = lstm(1, hidden, horizon)
early_count = 0
early_val = float('inf')
early_params = None
model_params = [model.Wf, model.Wc, model.Wi, model.Wo, model.Wy,model.bf, model.bc, model.bi, model.bo, model.by]

# train
for e in range(epochs):
    losses = []

    # shuffle la ferestre (ORDINEA DIN INT FERESTREI NU SE MODIFICA)
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]    
    y_train = y_train[p]

    # procesam datele pe batch-uri
    for i in range(0, len(x_train)-batch_size+1, batch_size):

        # luam cate un bacth de dim batch_size
        x_batch = torch.from_numpy(x_train[i : i+batch_size]).to(device)
        y_batch = torch.from_numpy(y_train[i : i+batch_size]).to(device)

        # obtinem predictiile
        y_hat = model.forward(x_batch)

        # calculcam loss-ul (mse)
        loss = torch.mean((y_batch - y_hat)**2)

        # backprop din librarii (autograd)
        loss.backward()

        # sgd (manual) 
        with torch.no_grad():
            for p in model_params:
                p -= lr * p.grad
                p.grad.zero_()

        losses.append(float(loss.item()))
    mse_train = float(np.mean(losses))


    # validare
    with torch.no_grad():
        x_batch = torch.from_numpy(x_val).to(device)
        y_batch = torch.from_numpy(y_val).to(device)
        y_hat = model.forward(x_batch)
        mse_val = float(torch.mean((y_batch - y_hat) ** 2).item())
    print(f"e {e+1} mse_train {mse_train} mse_val {mse_val}")


    # early stopping + checkpoint model
    if mse_val < early_val - delta:
        early_val = mse_val
        with torch.no_grad():
            early_params = [p.clone() for p in model_params]
        torch.save(early_params, "LSTM/results/lstm.pth")
        early_count = 0
    else:
        early_count += 1
        if early_count >= patience:
            break


# restaurare
saved_params = torch.load("LSTM/results/lstm.pth")
with torch.no_grad():
    for p, bp in zip(model_params, saved_params):
        p.copy_(bp)


# testare
start_win = x_train_all[-1]
start_win = (start_win - min_) / (max_ - min_ + delta)

with torch.no_grad():
    x_batch = torch.from_numpy(start_win.reshape((1, win_len, 1)).astype(np.float32)).to(device)
    y_hat = model.forward(x_batch)

# denormalizare
y_hat = y_hat.detach().cpu().numpy().reshape(-1)
y_hat = y_hat * (max_ - min_ + delta) + min_

y_true = y_test[0] * (max_ - min_ + delta) + min_

# rezultate
mse = float(np.mean((y_true - y_hat) ** 2))
print(f"Test MSE: {mse}")

plt.figure(figsize=(12,4))
plt.plot(y_true, label="true")
plt.plot(y_hat, label="pred")
plt.legend(); plt.grid(True)
plt.show()