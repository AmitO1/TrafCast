"""
evaluate.py – Model evaluation and prediction for traffic flow prediction

Features:
- Load trained model and encoder
- Generate predictions on new data
- Comprehensive evaluation metrics
- Visualization support
- Batch prediction for large datasets
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    print("Warning: seaborn not available, using matplotlib defaults")

from encode import TrafficDataEncoder
from train_lstm import LSTMRegressor


def load_model_and_encoder(
    model_path: str,
    encoder_path: str,
    device: torch.device
) -> Tuple[LSTMRegressor, TrafficDataEncoder]:
    """Load trained model and encoder."""
    print(f"Loading encoder from {encoder_path}")
    encoder = TrafficDataEncoder.load(encoder_path)
    
    print(f"Loading model from {model_path}")
    model_state = torch.load(model_path, map_location=device)
    
    # Infer model architecture from the saved state_dict
    n_features = len(encoder.num_cols) + len(encoder.cat_cols)
    
    # Infer hidden_size from the first LSTM layer weights
    # lstm.weight_ih_l0 shape is [4*hidden_size, n_features]
    first_layer_weight_shape = model_state['lstm.weight_ih_l0'].shape
    hidden_size = first_layer_weight_shape[0] // 4
    
    # Check if bidirectional by looking for reverse weights
    bidirectional = 'lstm.weight_ih_l0_reverse' in model_state
    
    # Infer number of layers by counting unique layer indices
    layer_keys = [k for k in model_state.keys() if k.startswith('lstm.weight_ih_l')]
    n_layers = len(set([k.split('_l')[1].split('_')[0] for k in layer_keys]))
    
    # Infer dropout from the model structure (this is harder to infer, so we'll use a default)
    dropout = 0.3  # Default value
    
    print(f"Inferred model architecture:")
    print(f"  n_features: {n_features}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  n_layers: {n_layers}")
    print(f"  bidirectional: {bidirectional}")
    print(f"  dropout: {dropout}")
    
    # Create model with inferred architecture
    model = LSTMRegressor(
        n_features=n_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        bidirectional=bidirectional
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    
    print("Model and encoder loaded successfully")
    return model, encoder


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Basic metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # Percentage metrics
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
            range_metrics[f'mae_{range_name.replace(" ", "_").replace("(", "").replace(")", "")}'] = np.mean(np.abs(range_pred - range_target))
            range_metrics[f'count_{range_name.replace(" ", "_").replace("(", "").replace(")", "")}'] = np.sum(mask)
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        **range_metrics
    }
    
    return metrics


def predict_batch(
    model: LSTMRegressor,
    encoder: TrafficDataEncoder,
    df: pd.DataFrame,
    batch_size: int = 256,
    device: torch.device = torch.device('cpu'),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions for the TEST portion of a dataset in batches.
    Uses the same chronological split as training to ensure we only evaluate on test data.
    
    Returns:
        predictions: (N,) - predicted values (test set only)
        targets: (N,) - actual values (test set only)
        target_indices: (N,) - indices of target rows in original df (test set only)
    """
    print("Encoding data for prediction...")
    X, y, target_indices, timestamps = encoder.transform(df)
    
    if len(X) == 0:
        print("No valid sequences found in data")
        return np.array([]), np.array([]), np.array([])
    
    # Apply the same chronological split as training
    print("Applying chronological split to match training...")
    sorted_indices = np.argsort(timestamps)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    target_indices_sorted = target_indices[sorted_indices]
    timestamps_sorted = timestamps[sorted_indices]
    
    # Calculate split points (same as training)
    n_total = len(X_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Get test indices
    test_indices = sorted_indices[n_train + n_val:]
    X_test = X[test_indices]
    y_test = y[test_indices]
    target_indices_test = target_indices[test_indices]
    
    print(f"Using test set: {len(X_test):,} samples ({(1-train_ratio-val_ratio)*100:.0f}%)")
    if len(X_test) > 0:
        test_timestamps = pd.to_datetime(timestamps[test_indices])
        print(f"Test date range: {test_timestamps.min()} to {test_timestamps.max()}")
    
    if len(X_test) == 0:
        print("No test data available")
        return np.array([]), np.array([]), np.array([])
    
    print(f"Generating predictions for {len(X_test)} test sequences...")
    
    # Create data loader for test set only
    dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_pred = model(batch_X).cpu().numpy()
            predictions.append(batch_pred)
            targets.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    
    return predictions, targets, target_indices_test


def create_evaluation_plots(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot: predictions vs targets
    axes[0, 0].scatter(targets, predictions, alpha=0.5, s=1)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Speed (mph)')
    axes[0, 0].set_ylabel('Predicted Speed (mph)')
    axes[0, 0].set_title('Predictions vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = predictions - targets
    axes[0, 1].scatter(targets, residuals, alpha=0.5, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Actual Speed (mph)')
    axes[0, 1].set_ylabel('Residuals (mph)')
    axes[0, 1].set_title('Residuals vs Actual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals (mph)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Speed range performance
    speed_ranges = {
        'Low (≤30)': targets <= 30,
        'Medium (30-60)': (targets > 30) & (targets <= 60),
        'High (≥60)': targets >= 60
    }
    
    range_maes = []
    range_names = []
    for name, mask in speed_ranges.items():
        if np.sum(mask) > 0:
            range_mae = np.mean(np.abs(predictions[mask] - targets[mask]))
            range_maes.append(range_mae)
            range_names.append(name)
    
    axes[1, 1].bar(range_names, range_maes, alpha=0.7)
    axes[1, 1].set_ylabel('MAE (mph)')
    axes[1, 1].set_title('MAE by Speed Range')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {save_path}")
    else:
        plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model")
    
    # Required arguments
    parser.add_argument("--csv", required=True, help="Path to CSV file with test data")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--encoder", required=True, help="Path to fitted encoder (.pkl file)")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for prediction")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training data ratio (must match training)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation data ratio (must match training)")
    parser.add_argument("--output", help="Path to save predictions CSV")
    parser.add_argument("--metrics_output", help="Path to save metrics JSON")
    parser.add_argument("--plots_output", help="Path to save evaluation plots")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model and encoder
    model, encoder = load_model_and_encoder(args.model, args.encoder, device)
    
    # Load test data
    print(f"Loading test data from {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows")
    
    # Generate predictions (using same split ratios as training)
    predictions, targets, target_indices = predict_batch(
        model, encoder, df, args.batch_size, device, 
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    
    if len(predictions) == 0:
        print("No predictions generated. Check your data format.")
        return
    
    # Compute metrics
    print("Computing evaluation metrics...")
    metrics = compute_metrics(predictions, targets)
    
    # Print metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"MAE (Mean Absolute Error): {metrics['mae']:.4f} mph")
    print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f} mph")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"R² (Coefficient of Determination): {metrics['r2']:.4f}")
    
    # Speed range metrics
    print("\nSpeed Range Performance:")
    for key, value in metrics.items():
        if key.startswith('mae_') and key.endswith('_count'):
            continue
        elif key.startswith('mae_'):
            range_name = key.replace('mae_', '').replace('_', ' ')
            count_key = f"count_{key.replace('mae_', '')}"
            count = metrics.get(count_key, 0)
            print(f"  {range_name.title()}: {value:.4f} mph (n={count})")
    
    # Save predictions if requested
    if args.output:
        print(f"\nSaving predictions to {args.output}")
        
        # Create detailed prediction DataFrame
        pred_df = pd.DataFrame({
            'prediction': predictions,
            'target': targets,
            'error': predictions - targets,
            'abs_error': np.abs(predictions - targets),
            'target_index': target_indices
        })
        
        # Add original data columns if possible
        if len(target_indices) > 0 and max(target_indices) < len(df):
            for col in ['Time', 'Latitude', 'Longitude', 'direction', 'weather']:
                if col in df.columns:
                    pred_df[col] = df.iloc[target_indices][col].values
        
        pred_df.to_csv(args.output, index=False)
        print(f"Predictions saved with {len(pred_df)} rows")
    
    # Save metrics if requested
    if args.metrics_output:
        import json

        # right before json.dump
        metrics = {k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else int(v) if isinstance(v, (np.integer,)) else v) 
                for k, v in metrics.items()}

        with open(args.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)

    
    # Create and save plots if requested
    if args.plots_output:
        print(f"Creating evaluation plots...")
        create_evaluation_plots(predictions, targets, args.plots_output)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
