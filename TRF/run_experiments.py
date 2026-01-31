import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchtst import PatchTST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


def train_model(config, x_train, y_train, x_val, y_val, exp_name):
    
    win_len = config['win_len']
    horizon = config['horizon']
    in_channels = config['in_channels']
    patch_len = config['patch_len']
    stride = config['stride']
    d_model = config['d_model']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    d_ff = config['d_ff']
    dropout = config['dropout']
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    patience = config['patience']
    delta = config['delta']
    
    print(f"Experiment: {exp_name}")
    print(f"win_len={win_len}, horizon={horizon}")
    
    #init model
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


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #early stopping
    early_count = 0
    early_val = float('inf')
    best_model_state = None
    best_epoch = 0
    
    #metrici
    train_losses = []
    val_losses = []
    val_maes = []
    
    #antrenare
    start_time = time.time()
    
    for e in range(epochs):
        epoch_start = time.time()
        losses = []
        
        #shuffle
        p = np.random.permutation(len(x_train))
        x_train_shuffled = x_train[p]
        y_train_shuffled = y_train[p]
        
        #procesare pe batch-uri
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
            loss = criterion(y_batch, y_hat)
            
            #backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #gradient clipping
            optimizer.step()
            
            losses.append(float(loss.item()))
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
        
        #clear cuda cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\n[Epoch {e+1}/{epochs}] Timp: {epoch_time:.2f}s")
        print(f"Train MSE: {mse_train:.6f} | Val MSE: {mse_val:.6f} | Val MAE: {mae_val:.6f}")
        
        #early stopping + checkpoint
        if mse_val < early_val - delta:
            early_val = mse_val
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = e + 1
            early_count = 0
            print(f"best model salvat (val_loss: {mse_val:.6f})")
        else:
            early_count += 1
            if early_count >= patience:
                print(f"\nearly stopping la epoca {e+1}")
                break
    
    #restaurare best model
    model.load_state_dict(best_model_state)
    print(f"\nbest model: epoca {best_epoch}")
    
    training_time = time.time() - start_time
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_maes': val_maes,
        'best_epoch': best_epoch,
        'training_time': training_time
    }


def evaluate_model(model, x_test, y_test): #evaluare pe test
    with torch.no_grad():
        x_batch = torch.from_numpy(x_test.astype(np.float32)).to(device)
        y_hat_all = model(x_batch)
    
    y_hat_all = y_hat_all.detach().cpu().numpy()
    y_true_all = y_test
    
    mse_test = mse(y_true_all, y_hat_all)
    mae_test = mae(y_true_all, y_hat_all)
    r2_test = r2_score(y_true_all, y_hat_all)
    mape_test = mape(y_true_all, y_hat_all)
    
    return {
        'mse': mse_test,
        'mae': mae_test,
        'r2': r2_test,
        'mape': mape_test
    }, y_hat_all


def save_plots(train_history, y_true, y_pred, exp_name, results_dir):
   
    #training history
    plt.figure(figsize=(12, 4))
    plt.plot(train_history['train_losses'], label="Train MSE")
    plt.plot(train_history['val_losses'], label="Val MSE")
    plt.xlabel("Epoca")
    plt.ylabel("MSE")
    plt.title(f"{exp_name} - Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{exp_name}_training_history.pdf"), format='pdf')
    plt.close()
    
    #prima fereastra de predictie
    plt.figure(figsize=(12, 4))
    y_true_plot = y_true[0].flatten()
    y_pred_plot = y_pred[0].flatten()
    plt.plot(y_true_plot, label="true")
    plt.plot(y_pred_plot, label="pred")
    plt.xlabel("timp")
    plt.ylabel("energie")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{exp_name}_test_prediction.pdf"), format='pdf')
    plt.close()


def main():
    #configuratii experimente
    experiments = [
        #set 1
        {'win_len': 168, 'horizon': 48, 'name': 'set1_wl168_h48'},
        {'win_len': 336, 'horizon': 48, 'name': 'set1_wl336_h48'},
        {'win_len': 504, 'horizon': 48, 'name': 'set1_wl504_h48'},
        
        #set 2
        {'win_len': 336, 'horizon': 24, 'name': 'set2_wl336_h24'},
        {'win_len': 336, 'horizon': 48, 'name': 'set2_wl336_h48'},
        {'win_len': 336, 'horizon': 72, 'name': 'set2_wl336_h72'},
    ]
    
    #hiperparametri comuni
    base_config = {
        'in_channels': 1,
        'patch_len': 16,
        'stride': 8,
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 3,
        'd_ff': 1024,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 20,
        'batch_size': 64,
        'patience': 5,
        'delta': 0.0001
    }
    
    #incarcarea datelor
    print("INCARCARE DATE")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "AEP_hourly.csv")
    data = pd.read_csv(data_path)
    data["time"] = pd.to_datetime(data["time"])
    data = data.set_index("time")
    energy = data["energy"].values
    print(f"Total date: {len(energy)} puncte")
    
    #split 80% train, 10% val, 10% test
    n = len(energy)
    train_data = energy[:int(n*0.8)]
    val_data = energy[int(n*0.8):int(n*0.9)]
    test_data = energy[int(n*0.9):]
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results", "experiments")
    os.makedirs(results_dir, exist_ok=True)
    

    rezultate = []
    
    #rulare experimente
    for exp_config in experiments:
        exp_name = exp_config['name']
        win_len = exp_config['win_len']
        horizon = exp_config['horizon']
        
        #pregatire date
        x_train, y_train = create_samples(train_data, win_len, horizon)
        x_val, y_val = create_samples(val_data, win_len, horizon)
        x_test, y_test = create_samples(test_data, win_len, horizon)
        
     
        x_train = x_train / 1000.0
        y_train = y_train / 1000.0
        x_val = x_val / 1000.0
        y_val = y_val / 1000.0
        x_test = x_test / 1000.0
        y_test = y_test / 1000.0
        
        x_train = x_train.reshape((-1, win_len, 1))
        x_val = x_val.reshape((-1, win_len, 1))
        x_test = x_test.reshape((-1, win_len, 1))
        y_train = y_train.reshape((-1, horizon, 1))
        y_val = y_val.reshape((-1, horizon, 1))
        y_test = y_test.reshape((-1, horizon, 1))
        
        config = {**base_config, 'win_len': win_len, 'horizon': horizon}
        
        #antrenare
        model, train_history = train_model(config, x_train, y_train, x_val, y_val, exp_name)
        
        #evaluare test
        print(f"EVALUARE TEST - {exp_name}")
        test_metrics, y_pred = evaluate_model(model, x_test, y_test)
        
        print(f"Test MSE:  {test_metrics['mse']:.6f}")
        print(f"Test MAE:  {test_metrics['mae']:.6f}")
        print(f"Test R2:   {test_metrics['r2']:.6f}")
        print(f"Test MAPE: {test_metrics['mape']:.2f}%")
        
        #salveaza grafice
        save_plots(train_history, y_test, y_pred, exp_name, results_dir)
        
        #salveaza modelul
        model_path = os.path.join(results_dir, f"{exp_name}_model.pth")
        torch.save(model.state_dict(), model_path)
        
        #colecteaza rezultate
        result = {
            'experiment': exp_name,
            'win_len': win_len,
            'horizon': horizon,
            'train_samples': len(x_train),
            'val_samples': len(x_val),
            'test_samples': len(x_test),
            'best_epoch': train_history['best_epoch'],
            'training_time_sec': train_history['training_time'],
            'final_train_mse': train_history['train_losses'][-1],
            'final_val_mse': train_history['val_losses'][-1],
            'test_mse': test_metrics['mse'],
            'test_mae': test_metrics['mae'],
            'test_r2': test_metrics['r2'],
            'test_mape': test_metrics['mape']
        }
        rezultate.append(result)
    
    #salveaza rezultatele in CSV
    results_df = pd.DataFrame(rezultate)
    results_csv = os.path.join(results_dir, "all_experiments_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"REZULTATE SALVATE: {results_csv}")
    print(results_df.to_string(index=False))
    
    #salveaza JSON
    results_json = os.path.join(results_dir, "all_experiments_results.json")
    with open(results_json, 'w') as f:
        json.dump(rezultate, f, indent=2)
    
    print(f"\nTOATE EXPERIMENTELE FINALIZATE!")
    print(f"rezultate salvate in: {results_dir}")


if __name__ == "__main__":
    main()
