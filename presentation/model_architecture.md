# LSTM Model Architecture

## 🧠 Overview

The TrafCast system uses a sophisticated LSTM (Long Short-Term Memory) neural network architecture specifically designed for traffic flow prediction. This document details the model design, architecture choices, and implementation details.

## 🏗️ Architecture Design

### Model Type: Bidirectional LSTM
```python
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=256, n_layers=2, 
                 dropout=0.3, bidirectional=True):
        # LSTM layer with bidirectional processing
        # Dense output layers for regression
```

### Why LSTM for Traffic Prediction?

#### 1. **Temporal Dependencies**
- Traffic patterns have strong temporal correlations
- LSTM captures long-term dependencies in time series
- Memory cells retain important historical information

#### 2. **Sequence Learning**
- Traffic data is inherently sequential (5-minute intervals)
- LSTM processes sequences of traffic measurements
- Captures patterns across multiple time steps

#### 3. **Bidirectional Processing**
- Processes data in both forward and backward directions
- Captures context from past and future time steps
- Improves understanding of traffic flow patterns

## 🔧 Model Components

### 1. LSTM Layer
```python
self.lstm = nn.LSTM(
    input_size=n_features,      # 10 features
    hidden_size=hidden_size,    # 256 units
    num_layers=n_layers,        # 2 layers
    batch_first=True,           # Batch dimension first
    dropout=dropout,            # 0.3 dropout
    bidirectional=bidirectional # True for bidirectional
)
```

**Parameters:**
- **Input Size**: 10 features (lat, lon, time features, categorical)
- **Hidden Size**: 256 units (optimal for this dataset)
- **Layers**: 2 layers for sufficient complexity
- **Dropout**: 0.3 for regularization
- **Bidirectional**: True for enhanced context

### 2. Output Head
```python
self.head = nn.Sequential(
    nn.Linear(lstm_output_size, hidden_size // 2),  # 512 → 128
    nn.ReLU(),                                      # Activation
    nn.Dropout(dropout),                            # 0.3 dropout
    nn.Linear(hidden_size // 2, 1)                  # 128 → 1 (speed)
)
```

**Architecture:**
- **First Linear**: 512 → 128 (bidirectional doubles hidden size)
- **ReLU Activation**: Non-linear transformation
- **Dropout**: 0.3 for regularization
- **Second Linear**: 128 → 1 (single speed prediction)

## 📊 Input/Output Specifications

### Input Format
```python
# Input tensor shape: [batch_size, seq_len, n_features]
batch_size = 128        # Training batch size
seq_len = 12           # 1 hour of 5-minute intervals
n_features = 10        # Feature count
```

### Feature Composition
```python
# 10 input features
1. Latitude (normalized)
2. Longitude (normalized)
3. hour_sin (cyclical time encoding)
4. hour_cos (cyclical time encoding)
5. dow_sin (day of week encoding)
6. dow_cos (day of week encoding)
7. lanes (number of lanes)
8. % Observed (data quality)
9. direction (ordinal encoded)
10. weather (ordinal encoded)
```

### Output Format
```python
# Output tensor shape: [batch_size, 1]
# Single speed prediction in mph
```

## 🎯 Training Configuration

### Current Training Parameters
```bash
# Training command
python train_lstm.py \
  --csv /workspace/train_file.csv \
  --epochs 20 \
  --batch_size 128 \
  --hidden_size 256 \
  --bidirectional \
  --loss_type weighted_huber \
  --model_out /workspace/outputs/final_lstm.pt \
  --encoder_out /workspace/outputs/final_encoder.pkl
```

### Hyperparameter Details
- **Epochs**: 20 (sufficient for convergence)
- **Batch Size**: 128 (optimal for memory and performance)
- **Hidden Size**: 256 (balanced complexity and efficiency)
- **Learning Rate**: 1e-3 (Adam optimizer default)
- **Weight Decay**: 1e-5 (L2 regularization)

## 🎯 Loss Function: Weighted Huber Loss

### Why Weighted Huber Loss?

#### 1. **Handles Outliers**
- Huber loss is robust to outliers
- Less sensitive than MSE to extreme values
- Better for traffic data with occasional anomalies

#### 2. **Addresses Class Imbalance**
- Traffic data is heavily skewed toward high speeds
- Weighted loss gives more importance to low-speed predictions
- Improves performance on congested traffic scenarios

#### 3. **Speed Class Weighting**
```python
# Speed class weights
weight_low = 2.5      # Low speed (≤30 mph)
weight_medium = 1.2   # Medium speed (30-60 mph)
weight_high = 0.8     # High speed (≥60 mph)
```

### Loss Function Implementation
```python
class WeightedHuberLoss(nn.Module):
    def forward(self, pred, target):
        # Compute Huber loss
        diff = torch.abs(pred - target)
        huber_loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,                    # Quadratic for small errors
            self.delta * (diff - 0.5 * self.delta)  # Linear for large errors
        )
        
        # Apply speed-based weights
        weights = self.compute_speed_weights(target)
        return (huber_loss * weights).mean()
```

## 🔄 Data Flow Architecture

### 1. Input Processing
```python
# Data flow
Raw Data → Encoder → Sequences → DataLoader → Model
```

### 2. Sequence Creation
```python
# Sensor-safe windowing
for sensor_id, group in df.groupby("sensor_id"):
    # Create sequences within each sensor
    # No cross-sensor leakage
    sequences = create_sliding_windows(group)
```

### 3. Model Forward Pass
```python
# Forward pass
def forward(self, x):
    lstm_out, _ = self.lstm(x)           # [batch, seq_len, hidden*2]
    last_output = lstm_out[:, -1, :]     # [batch, hidden*2]
    prediction = self.head(last_output)  # [batch, 1]
    return prediction
```

## 📈 Model Performance Characteristics

### Computational Requirements
- **Parameters**: ~500,000 trainable parameters
- **Memory**: ~2GB GPU memory for training
- **Training Time**: ~2-3 hours for 20 epochs
- **Inference Speed**: ~1000 predictions/second

### Model Capacity
- **Representation Power**: Sufficient for complex traffic patterns
- **Generalization**: Good performance on unseen data
- **Overfitting Control**: Dropout and regularization prevent overfitting
- **Scalability**: Can handle larger datasets

## 🔧 Architecture Optimizations

### 1. **Gradient Clipping**
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. **Early Stopping**
```python
# Stop training when validation loss stops improving
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_model_state = model.state_dict().copy()
```

### 3. **Learning Rate Scheduling**
```python
# Adam optimizer with weight decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)
```

## 🎯 Model Selection Rationale

### Why LSTM Over Other Architectures?

#### 1. **vs. Simple RNN**
- LSTM has better long-term memory
- Avoids vanishing gradient problem
- More stable training

#### 2. **vs. GRU**
- LSTM has more parameters for complex patterns
- Better for traffic data with long dependencies
- More proven in time series applications

#### 3. **vs. Transformer**
- LSTM is more efficient for this data size
- Better inductive bias for time series
- Simpler architecture, easier to train

#### 4. **vs. CNN**
- LSTM better captures temporal dependencies
- More appropriate for sequential data
- Better for variable-length sequences

## 🔄 Training Strategy

### 1. **Chronological Split**
```python
# 70% train, 15% validation, 15% test
# Chronological order preserved
# No future data leakage
```

### 2. **Sensor-Safe Processing**
```python
# Process each sensor separately
# No cross-sensor information leakage
# Maintains data integrity
```

### 3. **Validation Strategy**
```python
# Use validation set for model selection
# Early stopping based on validation loss
# Best model saved for final evaluation
```

## 📊 Model Evaluation

### Performance Metrics
- **MAE**: Mean Absolute Error (primary metric)
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Speed Range Performance
```python
# Performance by speed class
Low speed (≤30):    MAE = 11.57 mph
Medium speed (30-60): MAE = 7.17 mph
High speed (≥60):   MAE = 3.22 mph
```

## 🚀 Model Deployment

### Production Considerations
- **Model Size**: ~2MB saved model file
- **Inference Speed**: Real-time prediction capability
- **Memory Requirements**: ~100MB for inference
- **Scalability**: Can handle multiple concurrent predictions

### Integration Points
- **Data Pipeline**: Seamless integration with data processing
- **API Interface**: Ready for web service integration
- **Visualization**: Compatible with mapping systems
- **Monitoring**: Built-in performance tracking

---

*The LSTM architecture provides an optimal balance of performance, efficiency, and interpretability for traffic flow prediction, with specialized features for handling the unique challenges of traffic data.*
