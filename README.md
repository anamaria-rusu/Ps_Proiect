# Time Series Forecasting: LSTM/GRU vs Transformer Comparison

## Project Overview

This project implements and compares different deep learning architectures for time series forecasting on hourly energy consumption data. The main focus is on comparing traditional recurrent architectures (LSTM, GRU) with modern Transformer-based approaches (PatchTST) for long-term time series prediction.

## Models Implemented

### 1. LSTM (Long Short-Term Memory)
- Custom manual implementation from scratch (forward pass)
- Xavier weight initialization
- Gradient clipping for stability
- Configurable hidden layers and prediction horizons

### 2. GRU (Gated Recurrent Unit)
- Simplified recurrent architecture
- Fewer parameters than LSTM
- Faster training convergence

### 3. PatchTST (Patch Time Series Transformer)
- State-of-the-art Transformer-based architecture
- Implements patching mechanism for time series
- Reversible Instance Normalization (RevIN)
- Multi-head self-attention mechanism
- Positional encoding for temporal information
- Based on the paper "A Time Series is Worth 64 Words"

## Dataset

**AEP Hourly Energy Consumption Dataset**
- Source: American Electric Power (AEP)
- Time range: 2004-2018
- Frequency: Hourly measurements
- Total records: 121,273 hourly observations
- Features: Timestamp and energy consumption (MW)

## Key Features

- **Multiple Window Lengths**: Experiments with different input sequence lengths (168, 336, 504 hours)
- **Variable Prediction Horizons**: Testing different forecast horizons (24, 48, 72 hours)
- **Comprehensive Metrics**: MSE, MAE, MAPE, R² score
- **Training Visualization**: Loss curves and prediction plots
- **Model Persistence**: Save/load trained models
- **GPU Acceleration**: CUDA support for faster training

## Project Structure

```
Ps_Proiect/
├── data/
│   └── AEP_hourly.csv           # Energy consumption dataset
├── LSTM/
│   ├── lstm.py                  # Custom LSTM implementation
│   └── results/
│       ├── models/              # Saved LSTM models
│       └── plots/               # Training visualizations
├── GRU/
│   └── __init__.py              # GRU implementation
├── TRF/
│   ├── patchtst.py             # PatchTST model architecture
│   ├── encoder.py              # Transformer encoder
│   ├── attention.py            # Multi-head attention mechanism
│   ├── embeddings.py           # Patch embedding & positional encoding
│   ├── layers.py               # RevIN and other layers
│   ├── train_patchtst.py       # Training script
│   ├── run_experiments.py      # Automated experiments
│   ├── visualize_positional_encoding.py
│   └── results/
│       └── experiments/        # Experiment results and models
└── bibl PS/
    └── diagrama_transformer.drawio  # Architecture diagrams
```

## Installation

### Requirements
```bash
Python 3.8+
numpy
pandas
matplotlib
torch
tqdm
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Ps_Proiect

# Create virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy pandas matplotlib torch tqdm
```

## Usage

### Training LSTM Model
```bash
cd LSTM
python lstm.py
```

### Training PatchTST (Transformer)
```bash
cd TRF
python train_patchtst.py
```

### Running Comprehensive Experiments
```bash
cd TRF
python run_experiments.py
```

This will run multiple experiments with different configurations:
- **Set 1**: Fixed horizon (48h), varying window lengths (168, 336, 504 hours)
- **Set 2**: Fixed window length (336h), varying horizons (24, 48, 72 hours)

## Model Architecture Details

### PatchTST Architecture
1. **Input**: (Batch, Sequence Length, Channels)
2. **Reversible Instance Normalization**: Stabilize training
3. **Patching**: Divide time series into overlapping patches
4. **Linear Projection**: Map patches to embedding space
5. **Positional Encoding**: Add temporal position information
6. **Transformer Encoder**: Multi-layer self-attention
7. **Prediction Head**: Map to forecast horizon
8. **Denormalization**: Restore original scale

### LSTM Architecture
- **Input Gate**: Controls information flow into cell state
- **Forget Gate**: Decides what information to discard
- **Cell State**: Long-term memory
- **Output Gate**: Controls hidden state output
- **Prediction Layer**: Linear projection to forecast

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **MAE (Mean Absolute Error)**: Average absolute difference
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric
- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model

## Experimental Results

All experiments were conducted using the PatchTST (Transformer) model on the AEP hourly energy dataset with a 80/10/10 train/validation/test split.

### Experiment Set 1: Effect of Window Length (Fixed Horizon = 48 hours)

| Window Length | MSE | MAE | R² | MAPE (%) | Training Time (s) | Best Epoch |
|---------------|------|------|------|----------|-------------------|------------|
| **168 hours** (1 week) | 0.933 | 0.686 | **0.846** | **4.61** | 559 | 18 |
| **336 hours** (2 weeks) | 0.955 | 0.706 | 0.843 | 4.79 | 622 | 14 |
| **504 hours** (3 weeks) | 1.034 | 0.737 | 0.830 | 4.99 | 845 | 15 |

**Key Findings:**
- Shorter window length (168h) achieved the best performance with lowest MSE (0.933) and highest R² (0.846)
- MAPE remained consistently low (< 5%) across all window lengths
- Longer windows require more training time but don't necessarily improve accuracy
- Sweet spot appears to be around 1-2 weeks of historical data for 48-hour forecasts

### Experiment Set 2: Effect of Forecast Horizon (Fixed Window = 336 hours)

| Forecast Horizon | MSE | MAE | R² | MAPE (%) | Training Time (s) | Best Epoch |
|------------------|------|------|------|----------|-------------------|------------|
| **24 hours** (1 day) | **0.649** | **0.590** | **0.894** | **4.02** | 534 | 14 |
| **48 hours** (2 days) | 0.980 | 0.711 | 0.839 | 4.80 | 556 | 19 |
| **72 hours** (3 days) | 1.402 | 0.861 | 0.768 | 5.78 | 530 | 14 |

**Key Findings:**
- Performance degrades significantly as forecast horizon increases
- 24-hour forecasts achieve excellent R² of 0.894 with MAPE of only 4.02%
- MSE more than doubles when extending from 24h to 72h forecasts
- Model maintains good performance (R² > 0.76) even for 3-day forecasts
- All experiments converge relatively quickly (14-19 epochs)

### Overall Performance Summary

**Best Configuration:**
- **Window Length**: 336 hours (2 weeks)
- **Forecast Horizon**: 24 hours (1 day)
- **Performance**: R² = 0.894, MAPE = 4.02%

**Observations:**
1. The model shows strong predictive capability with R² scores above 0.76 for all configurations
2. MAPE values remain under 6% across all experiments, indicating high practical accuracy
3. Training times are reasonable (8-14 minutes on GPU) for all configurations
4. Short-term forecasts (24h) are significantly more accurate than long-term (72h)
5. The model generalizes well with minimal overfitting (validation MSE close to test MSE)

### LSTM Model Results

The custom LSTM implementation was tested on the same configurations for direct comparison:

| Configuration | MSE | MAE | R² | MAPE (%) |
|---------------|------|------|------|----------|
| **24h horizon, 336h window** | 0.547 | 0.513 | **0.910** | **3.49** |
| **48h horizon, 168h window** | 0.858 | 0.659 | 0.858 | 4.47 |
| **48h horizon, 336h window** | 0.822 | 0.638 | 0.865 | 4.34 |
| **48h horizon, 504h window** | 0.843 | 0.654 | 0.861 | 4.48 |
| **72h horizon, 336h window** | 1.202 | 0.791 | 0.801 | 5.35 |

### Model Comparison: PatchTST vs LSTM

Direct performance comparison on identical configurations:

#### 24-Hour Forecast (336h window)
| Model | MSE | MAE | R² | MAPE (%) |
|-------|------|------|------|----------|
| **LSTM** | **0.547** | **0.513** | **0.910** | **3.49** |
| **PatchTST** | 0.649 | 0.590 | 0.894 | 4.02 |

**Winner:** LSTM performs slightly better on 24h forecasts

#### 48-Hour Forecast (168h window)
| Model | MSE | MAE | R² | MAPE (%) |
|-------|------|------|------|----------|
| **LSTM** | 0.858 | 0.659 | 0.858 | 4.47 |
| **PatchTST** | **0.933** | **0.686** | 0.846 | 4.61 |

**Winner:** Comparable performance, LSTM edge on short windows

#### 48-Hour Forecast (336h window)
| Model | MSE | MAE | R² | MAPE (%) |
|-------|------|------|------|----------|
| **LSTM** | **0.822** | **0.638** | **0.865** | **4.34** |
| **PatchTST** | 0.980 | 0.711 | 0.839 | 4.80 |

**Winner:** LSTM shows better performance

#### 48-Hour Forecast (504h window)
| Model | MSE | MAE | R² | MAPE (%) |
|-------|------|------|------|----------|
| **LSTM** | **0.843** | **0.654** | **0.861** | **4.48** |
| **PatchTST** | 1.034 | 0.737 | 0.830 | 4.99 |

**Winner:** LSTM maintains advantage on longer windows

#### 72-Hour Forecast (336h window)
| Model | MSE | MAE | R² | MAPE (%) |
|-------|------|------|------|----------|
| **LSTM** | **1.202** | **0.791** | **0.801** | **5.35** |
| **PatchTST** | 1.402 | 0.861 | 0.768 | 5.78 |

**Winner:** LSTM performs better on longer forecasts

### Key Insights from Model Comparison

**LSTM Advantages:**
- **Better overall accuracy** on this specific energy forecasting task
- **Lower error rates** across all tested configurations (5-16% lower MSE)
- **Superior short-term predictions** especially for 24h forecasts (R² = 0.910)
- **More efficient** for univariate time series with clear sequential patterns
- **Consistent performance** across varying window lengths

**PatchTST Advantages:**
- **Faster training** through parallel processing of patches
- **Better scalability** for multivariate and high-dimensional data
- **Competitive performance** with R² > 0.76 across all configurations
- **Architectural flexibility** with attention mechanisms
- **State-of-the-art design** with modern techniques (RevIN, patching)

**Overall Conclusion:**
For this specific univariate energy forecasting task, the custom LSTM implementation outperforms PatchTST across all configurations. This demonstrates that:
1. **Simpler models can outperform complex ones** when the task matches the model's strengths
2. **Sequential recurrent architectures** are well-suited for univariate time series with strong temporal dependencies
3. **Transformer-based models** excel more in scenarios with multivariate data or when capturing long-range dependencies across multiple features
4. **Both models achieve practical accuracy** with MAPE < 6% for all forecasts

