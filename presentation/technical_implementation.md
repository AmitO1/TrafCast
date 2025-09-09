# Technical Implementation

## 🏗️ System Architecture

### Overview
The TrafCast system is built with a modular architecture that separates data collection, processing, modeling, and visualization components. This design ensures scalability, maintainability, and ease of deployment.

### Core Components
```
TrafCast System Architecture:
├── Data Collection Module
├── Data Processing Pipeline
├── LSTM Model Training
├── Road Network System
├── Visualization Engine
└── Deployment Interface
```

## 📁 Project Structure

### Directory Organization
```
TrafCast/
├── data_collection/          # Automated data gathering
│   ├── collect.py           # Web scraping automation
│   ├── data/               # Raw traffic data
│   └── coordinates/        # Highway coordinate files
├── data_process/           # Data preprocessing
│   ├── process.py         # Data cleaning and preparation
│   ├── split.py           # Data splitting utilities
│   └── unified.py         # Data unification
├── model_v3/              # Latest model implementation
│   ├── train_lstm.py      # Model training
│   ├── encode.py          # Data encoding
│   ├── evaluate.py        # Model evaluation
│   └── predict_road.py    # Road-specific predictions
├── roadmap/               # Geographic system
│   ├── RoadMap.py         # Road network management
│   ├── utils.py           # Geographic utilities
│   └── mock_predictor.py  # Mock prediction system
└── presentation/          # Project documentation
    └── [presentation files]
```

## 🔧 Technology Stack

### Core Technologies
```python
# Deep Learning
- PyTorch: Neural network framework
- torch.nn: LSTM implementation
- torch.optim: Optimization algorithms

# Data Processing
- pandas: Data manipulation
- numpy: Numerical computations
- scikit-learn: Preprocessing and utilities

# Geographic Processing
- osmnx: OpenStreetMap integration
- geopandas: Geographic data handling
- folium: Interactive mapping
- shapely: Geometric operations

# Web Scraping
- selenium: Automated web interaction
- requests: HTTP requests
- BeautifulSoup: HTML parsing

# Visualization
- matplotlib: Static plots
- seaborn: Statistical visualization
- folium: Interactive maps
- plotly: Dynamic visualizations
```

### Development Tools
```python
# Development Environment
- Python 3.8+
- Jupyter Notebooks
- VS Code / PyCharm
- Git version control

# Deployment
- Docker containers
- REST API framework
- Cloud deployment ready
- Monitoring and logging
```

## 🚀 Data Collection Implementation

### Web Scraping Automation
```python
# Selenium-based automation
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Chrome configuration
chrome_options = Options()
chrome_prefs = {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
}
chrome_options.add_experimental_option("prefs", chrome_prefs)

# Automated data collection
driver = webdriver.Chrome(service=Service('/opt/homebrew/bin/chromedriver'), 
                         options=chrome_options)
```

### Data Collection Features
1. **Automated Login**: Credential-based authentication
2. **Systematic Coverage**: Iterates through all target highways
3. **Directional Coverage**: Collects data for both directions
4. **Temporal Coverage**: Downloads full month of data
5. **File Management**: Automatic naming and organization

## 🔄 Data Processing Pipeline

### TrafficDataEncoder Class
```python
class TrafficDataEncoder:
    def __init__(self, seq_len=12, horizon=1, target_col="speed_mph"):
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_col = target_col
        
        # Feature columns
        self.cat_cols = ["direction", "weather"]
        self.num_cols = [
            "lanes", "% Observed", "Latitude", "Longitude",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ]
```

### Key Processing Steps
1. **Sensor ID Creation**: Unique identifier from coordinates
2. **Time Feature Engineering**: Cyclical encoding
3. **Geographic Integration**: Coordinate matching
4. **Sequence Creation**: Sensor-safe windowing
5. **Data Validation**: Quality checks and filtering

### Feature Engineering
```python
# Cyclical time encoding
def _add_time_features(self, df):
    dt = pd.to_datetime(df["Time"], errors="coerce")
    hour = dt.dt.hour + dt.dt.minute / 60.0
    dow = dt.dt.dayofweek
    
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    return df
```

## 🧠 LSTM Model Implementation

### Model Architecture
```python
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=256, n_layers=2, 
                 dropout=0.3, bidirectional=True):
        super().__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
```

### Training Implementation
```python
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

## 🗺️ Road Network System

### RoadMapManager Class
```python
class RoadMapManager:
    def __init__(self, city, bbox, base_data_dir="data"):
        self.city = city
        self.bbox = bbox
        self.base_data_dir = base_data_dir
        self.city_path = os.path.join(self.base_data_dir, self.city)
        
        # Directory structure
        self.coordinates_path = os.path.join(self.city_path, 'coordinates')
        self.predictions_path = os.path.join(self.city_path, 'predictions')
        self.road_network_path = os.path.join(self.city_path, 'maps')
        self.visualizations_path = os.path.join(self.city_path, 'visualizations')
        
        self._load_road_network(self.bbox)
```

### Geographic Processing
```python
# OSMnx integration
def _load_road_network(self, bbox):
    network_filename = f"{self.city.replace(' ', '_')}_network.graphml"
    network_path = os.path.join(self.road_network_path, network_filename)
    
    if os.path.exists(network_path):
        self.road_network = ox.load_graphml(network_path)
    else:
        self.road_network = ox.graph_from_bbox(
            bbox=bbox,
            network_type='drive'
        )
        ox.save_graphml(self.road_network, filepath=network_path)
```

### Coordinate Matching
```python
def add_coordinate(df_coord, df_data):
    # Sort by postmile for efficient matching
    df_coord = df_coord.sort_values(by="Abs PM").reset_index(drop=True)
    df_data = df_data.sort_values(by="Postmile (Abs)").reset_index(drop=True)
    
    # Find closest coordinate for each postmile
    coord_abs_pm = df_coord["Abs PM"].values
    coord_lat = df_coord["Latitude"].values
    coord_lon = df_coord["Longitude"].values
    
    def find_closest_index(target):
        return np.abs(coord_abs_pm - target).argmin()
    
    closest_indices = df_data["Postmile (Abs)"].apply(find_closest_index)
    df_data["Latitude"] = closest_indices.apply(lambda idx: coord_lat[idx])
    df_data["Longitude"] = closest_indices.apply(lambda idx: coord_lon[idx])
    
    return df_data
```

## 🎨 Visualization System

### Interactive Mapping
```python
def draw_map_offset(self):
    # Create dark-themed map
    center_lon = (self.bbox[0] + self.bbox[2]) / 2
    center_lat = (self.bbox[1] + self.bbox[3]) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='CartoDB dark_matter'
    )
    
    # Color coding based on speed
    def get_color(speed, max_speed):
        if speed >= 0.85 * max_speed:
            return '#00FF00'  # Green - free flow
        elif speed >= 0.55 * max_speed:
            return '#FFA500'  # Orange - moderate
        else:
            return '#FF0000'  # Red - congested
    
    # Add traffic segments
    for (road_name, direction), df in self.roads.items():
        for i in range(len(df) - 1):
            lat1, lon1, speed1 = df.loc[i, ['Latitude', 'Longitude', 'speed']]
            lat2, lon2, speed2 = df.loc[i+1, ['Latitude', 'Longitude', 'speed']]
            
            avg_speed = (speed1 + speed2) / 2
            color = get_color(avg_speed, max_speed)
            
            folium.PolyLine(
                locations=[(lat1, lon1), (lat2, lon2)],
                color=color,
                weight=2,
                opacity=0.95
            ).add_to(m)
    
    return m
```

## 🔧 Loss Function Implementation

### Weighted Huber Loss
```python
class WeightedHuberLoss(nn.Module):
    def __init__(self, weight_dict, delta=1.0, boost_low=1.0):
        super().__init__()
        self.delta = delta
        self.weight_low = weight_dict["weight_low"] * boost_low
        self.weight_medium = weight_dict["weight_medium"]
        self.weight_high = weight_dict["weight_high"]
        self.low_threshold = weight_dict["low_threshold"]
        self.high_threshold = weight_dict["high_threshold"]
    
    def forward(self, pred, target):
        # Compute Huber loss
        diff = torch.abs(pred - target)
        huber_loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        # Apply speed-based weights
        weights = torch.ones_like(target)
        low_mask = target <= self.low_threshold
        high_mask = target >= self.high_threshold
        medium_mask = ~(low_mask | high_mask)
        
        weights[low_mask] = self.weight_low
        weights[medium_mask] = self.weight_medium
        weights[high_mask] = self.weight_high
        
        return (huber_loss * weights).mean()
```

## 📊 Evaluation System

### Comprehensive Metrics
```python
def compute_metrics(predictions, targets):
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Basic metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Speed-specific metrics
    speed_ranges = {
        'low (≤30)': targets <= 30,
        'medium (30-60)': (targets > 30) & (targets <= 60),
        'high (≥60)': targets >= 60
    }
    
    range_metrics = {}
    for range_name, mask in speed_ranges.items():
        if np.sum(mask) > 0:
            range_pred = predictions[mask]
            range_target = targets[mask]
            range_metrics[f'mae_{range_name.replace(" ", "_")}'] = np.mean(np.abs(range_pred - range_target))
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'r2': r2,
        **range_metrics
    }
```

## 🚀 Deployment Architecture

### Model Serving
```python
# Model loading and inference
def load_model_and_encoder(model_path, encoder_path, device):
    encoder = TrafficDataEncoder.load(encoder_path)
    model_state = torch.load(model_path, map_location=device)
    
    # Infer model architecture
    n_features = len(encoder.num_cols) + len(encoder.cat_cols)
    hidden_size = model_state['lstm.weight_ih_l0'].shape[0] // 4
    bidirectional = 'lstm.weight_ih_l0_reverse' in model_state
    
    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=hidden_size,
        bidirectional=bidirectional
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    return model, encoder
```

### API Interface
```python
# Prediction endpoint
def predict_traffic(data, model, encoder):
    # Preprocess data
    X, y, _, _ = encoder.transform(data)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(X).float())
    
    return predictions.numpy()
```

## 🔧 Configuration Management

### Command Line Interface
```python
# Training configuration
parser = argparse.ArgumentParser(description="Train LSTM model for traffic prediction")

# Data parameters
parser.add_argument("--csv", required=True, help="Path to CSV file with traffic data")
parser.add_argument("--seq_len", type=int, default=12, help="Sequence length")
parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")

# Model parameters
parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")
parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")

# Training parameters
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

# Loss parameters
parser.add_argument("--loss_type", choices=["mse", "mae", "huber", "weighted_huber"], 
                   default="weighted_huber", help="Loss function type")

# Output parameters
parser.add_argument("--model_out", help="Path to save the best model")
parser.add_argument("--encoder_out", help="Path to save the fitted encoder")
```

## 📈 Performance Monitoring

### Training Monitoring
```python
# Training progress tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(1, args.epochs + 1):
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch:3d}/{args.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
```

### Logging and Debugging
```python
# Comprehensive logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

## 🔄 Data Pipeline Integration

### End-to-End Processing
```python
# Complete data pipeline
def process_traffic_data(raw_data_path, output_path):
    # 1. Load raw data
    df = pd.read_csv(raw_data_path)
    
    # 2. Create encoder
    encoder = TrafficDataEncoder(seq_len=12, horizon=1)
    
    # 3. Transform data
    X, y, target_indices, timestamps = encoder.fit_transform(df)
    
    # 4. Create data loaders
    train_loader, val_loader, test_loader, test_indices = create_data_loaders(
        X, y, timestamps, batch_size=128
    )
    
    # 5. Train model
    model = train_model(train_loader, val_loader)
    
    # 6. Evaluate model
    metrics = evaluate_model(model, test_loader)
    
    # 7. Save artifacts
    torch.save(model.state_dict(), f"{output_path}/model.pt")
    encoder.save(f"{output_path}/encoder.pkl")
    
    return model, encoder, metrics
```

---

*The technical implementation provides a robust, scalable, and maintainable foundation for the TrafCast traffic prediction system, with clear separation of concerns and comprehensive functionality.*
