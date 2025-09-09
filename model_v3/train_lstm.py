"""
train_lstm.py – LSTM model training for traffic flow prediction

Features:
- LSTM model with configurable architecture
- Weighted loss for handling speed class imbalance
- Huber loss option (better than regular loss per user experience)
- CLI interface for hyperparameter tuning
- Model and encoder saving
- Chronological train/val/test splits
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib
from typing import Dict, Tuple, Optional

from encode import TrafficDataEncoder


# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


class LSTMRegressor(nn.Module):
    """LSTM model for traffic speed prediction."""
    
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
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
    
    def forward(self, x):
        """Forward pass through the LSTM."""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        prediction = self.head(last_output)
        return prediction


class WeightedHuberLoss(nn.Module):
    """Weighted Huber loss for handling speed class imbalance."""
    
    def __init__(self, weight_dict: Dict[str, float], delta: float = 1.0, boost_low: float = 1.0):
        super().__init__()
        self.delta = delta
        self.weight_low = weight_dict["weight_low"] * boost_low  # Additional boost for low speeds
        self.weight_medium = weight_dict["weight_medium"]
        self.weight_high = weight_dict["weight_high"]
        self.low_threshold = weight_dict["low_threshold"]
        self.high_threshold = weight_dict["high_threshold"]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted Huber loss."""
        # Ensure target is 1D
        if target.dim() > 1:
            target = target.squeeze()
        if pred.dim() > 1:
            pred = pred.squeeze()
        
        # Compute Huber loss
        diff = torch.abs(pred - target)
        huber_loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        # Compute weights based on speed classes
        weights = torch.ones_like(target)
        low_mask = target <= self.low_threshold
        high_mask = target >= self.high_threshold
        medium_mask = ~(low_mask | high_mask)
        
        weights[low_mask] = self.weight_low
        weights[medium_mask] = self.weight_medium
        weights[high_mask] = self.weight_high
        
        # Apply weights
        weighted_loss = huber_loss * weights
        return weighted_loss.mean()


class FocalHuberLoss(nn.Module):
    """Focal loss variant for Huber loss to focus on hard examples."""
    
    def __init__(self, weight_dict: Dict[str, float], delta: float = 1.0, alpha: float = 2.0, gamma: float = 2.0):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.weight_low = weight_dict["weight_low"]
        self.weight_medium = weight_dict["weight_medium"]
        self.weight_high = weight_dict["weight_high"]
        self.low_threshold = weight_dict["low_threshold"]
        self.high_threshold = weight_dict["high_threshold"]
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal Huber loss."""
        if target.dim() > 1:
            target = target.squeeze()
        if pred.dim() > 1:
            pred = pred.squeeze()
        
        # Compute Huber loss
        diff = torch.abs(pred - target)
        huber_loss = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        # Compute focal weights (higher loss = harder example)
        focal_weights = self.alpha * (huber_loss ** self.gamma)
        
        # Apply class weights
        class_weights = torch.ones_like(target)
        low_mask = target <= self.low_threshold
        high_mask = target >= self.high_threshold
        medium_mask = ~(low_mask | high_mask)
        
        class_weights[low_mask] = self.weight_low
        class_weights[medium_mask] = self.weight_medium
        class_weights[high_mask] = self.weight_high
        
        # Combine focal and class weights
        total_weights = focal_weights * class_weights
        weighted_loss = huber_loss * total_weights
        
        return weighted_loss.mean()


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Create chronological train/validation/test data loaders.
    
    Args:
        X: Input sequences (N, seq_len, n_features)
        y: Target values (N, horizon)
        timestamps: Timestamps for each sample
        batch_size: Batch size for data loaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
    
    Returns:
        train_loader, val_loader, test_loader, test_indices
    """
    # Sort by timestamp to ensure chronological order
    sorted_indices = np.argsort(timestamps)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Calculate split points
    n_total = len(X_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split indices
    train_indices = sorted_indices[:n_train]
    val_indices = sorted_indices[n_train:n_train + n_val]
    test_indices = sorted_indices[n_train + n_val:]
    
    # Convert timestamps to datetime for date range display
    timestamps_dt = pd.to_datetime(timestamps)
    
    print(f"Data split:")
    print(f"  Train: {len(train_indices):,} samples ({train_ratio*100:.0f}%)")
    if len(train_indices) > 0:
        train_dates = timestamps_dt[train_indices]
        print(f"    Date range: {train_dates.min()} to {train_dates.max()}")
    
    print(f"  Val:   {len(val_indices):,} samples ({val_ratio*100:.0f}%)")
    if len(val_indices) > 0:
        val_dates = timestamps_dt[val_indices]
        print(f"    Date range: {val_dates.min()} to {val_dates.max()}")
    
    print(f"  Test:  {len(test_indices):,} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    if len(test_indices) > 0:
        test_dates = timestamps_dt[test_indices]
        print(f"    Date range: {test_dates.min()} to {test_dates.max()}")
    
    # Create data loaders
    def create_loader(indices, shuffle=False):
        X_subset = torch.from_numpy(X[indices]).float()
        y_subset = torch.from_numpy(y[indices]).float()
        dataset = TensorDataset(X_subset, y_subset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    train_loader = create_loader(train_indices, shuffle=True)
    val_loader = create_loader(val_indices, shuffle=False)
    test_loader = create_loader(test_indices, shuffle=False)
    
    return train_loader, val_loader, test_loader, test_indices


def train_epoch(
    model: LSTMRegressor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """Train the model for one epoch."""
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


def evaluate(
    model: LSTMRegressor,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> float:
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X)
            loss = loss_fn(predictions, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train LSTM model for traffic prediction")
    
    # Data parameters
    parser.add_argument("--csv", required=True, help="Path to CSV file with traffic data")
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length (default: 12)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (default: 1)")
    parser.add_argument("--target_col", default="speed_mph", help="Target column name")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    
    # Loss parameters
    parser.add_argument("--loss_type", choices=["mse", "mae", "huber", "weighted_huber", "focal_huber"], 
                       default="weighted_huber", help="Loss function type")
    parser.add_argument("--huber_delta", type=float, default=1.0, help="Huber loss delta")
    parser.add_argument("--boost_low", type=float, default=1.0, help="Additional boost for low-speed loss (weighted_huber only)")
    parser.add_argument("--focal_alpha", type=float, default=2.0, help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    
    # Data split parameters
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation data ratio")
    
    # Output parameters
    parser.add_argument("--model_out", help="Path to save the best model")
    parser.add_argument("--encoder_out", help="Path to save the fitted encoder")
    parser.add_argument("--pred_csv", help="Path to save test predictions")
    parser.add_argument("--log_file", help="Path to save training log")
    
    args = parser.parse_args()
    
    # Load and encode data
    print("Loading data...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows from {args.csv}")
    
    # Create encoder
    encoder = TrafficDataEncoder(
        seq_len=args.seq_len,
        horizon=args.horizon,
        target_col=args.target_col
    )
    
    # Fit encoder and transform data
    print("Encoding data...")
    X, y, target_indices, timestamps = encoder.fit_transform(df)
    print(f"Encoded data shapes: X={X.shape}, y={y.shape}")
    
    # Save encoder if requested
    if args.encoder_out:
        encoder.save(args.encoder_out)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, test_indices = create_data_loaders(
        X, y, timestamps, args.batch_size, args.train_ratio, args.val_ratio
    )
    
    # Initialize model
    print("Initializing model...")
    model = LSTMRegressor(
        n_features=X.shape[2],
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize loss function
    if args.loss_type == "weighted_huber":
        # Get speed weights from encoder
        weight_dict = encoder.get_speed_weights(y.flatten())
        loss_fn = WeightedHuberLoss(weight_dict, args.huber_delta, args.boost_low)
        print(f"Using weighted Huber loss with low-speed boost: {args.boost_low}")
    elif args.loss_type == "focal_huber":
        # Get speed weights from encoder
        weight_dict = encoder.get_speed_weights(y.flatten())
        loss_fn = FocalHuberLoss(weight_dict, args.huber_delta, args.focal_alpha, args.focal_gamma)
        print(f"Using focal Huber loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    elif args.loss_type == "huber":
        loss_fn = nn.SmoothL1Loss(beta=args.huber_delta)
        print("Using Huber loss")
    elif args.loss_type == "mae":
        loss_fn = nn.L1Loss()
        print("Using MAE loss")
    else:  # mse
        loss_fn = nn.MSELoss()
        print("Using MSE loss")
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        
        # Validate
        val_loss = evaluate(model, val_loader, loss_fn, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:3d}/{args.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  -> New best validation loss: {best_val_loss:.4f}")
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(best_model_state)
    test_loss = evaluate(model, test_loader, loss_fn, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save best model
    if args.model_out:
        torch.save(best_model_state, args.model_out)
        print(f"Best model saved to {args.model_out}")
    
    # Save predictions if requested
    if args.pred_csv:
        print("Generating test predictions...")
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE)
                batch_pred = model(batch_X).cpu().numpy()
                predictions.append(batch_pred)
                targets.append(batch_y.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'prediction': predictions.flatten(),
            'target': targets.flatten(),
            'error': predictions.flatten() - targets.flatten(),
            'abs_error': np.abs(predictions.flatten() - targets.flatten())
        })
        
        pred_df.to_csv(args.pred_csv, index=False)
        print(f"Predictions saved to {args.pred_csv}")
        
        # Print some statistics
        mae = pred_df['abs_error'].mean()
        rmse = np.sqrt((pred_df['error'] ** 2).mean())
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
    
    # Save training log if requested
    if args.log_file:
        log_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        log_df.to_csv(args.log_file, index=False)
        print(f"Training log saved to {args.log_file}")


if __name__ == "__main__":
    main()


