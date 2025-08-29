# train_lstm.py – save best model + device support + rich export + memory-efficient weighted loss
# ---------------------------------------------------------------------------
# Adds --model_out option to store the best-performing state_dict.
# Adds memory-efficient weighted loss to handle class imbalance in speed data.
# Example:
#   python model/train_lstm.py --csv data_process/example.csv \
#          --epochs 8 --batch 512 --model_out lstm.pt --pred_csv test_preds.csv \
#          --use_weighted_loss
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encode import TrafficDataEncoder

# ───────────────────────────── device selection ────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# ───────────────────────────── model definition ────────────────────────────────
class LSTMReg(nn.Module):
    def __init__(self, n_feats: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # use last timestep

# ───────────────────────────── memory-efficient weighted loss ──────────────────

def compute_speed_class_weights(y, speed_thresholds=(35, 55)):
    """
    Compute class weights for speed-based weighted loss (memory efficient).
    
    Args:
        y: target speed values
        speed_thresholds: (low_threshold, high_threshold) for speed classes
        
    Returns:
        weight_dict: dictionary with weights for each speed class
    """
    low_thresh, high_thresh = speed_thresholds
    
    # Define speed classes
    low_speed = y <= low_thresh
    medium_speed = (y > low_thresh) & (y <= high_thresh)
    high_speed = y > high_thresh
    
    # Count samples in each class
    n_low = low_speed.sum()
    n_medium = medium_speed.sum()
    n_high = high_speed.sum()
    n_total = len(y)
    
    print(f"Speed distribution:")
    print(f"  Low speed (≤{low_thresh}): {n_low} samples ({n_low/n_total*100:.1f}%)")
    print(f"  Medium speed ({low_thresh}-{high_thresh}): {n_medium} samples ({n_medium/n_total*100:.1f}%)")
    print(f"  High speed (>{high_thresh}): {n_high} samples ({n_high/n_total*100:.1f}%)")
    
    # Compute inverse frequency weights (normalized so average = 1)
    if n_low > 0 and n_medium > 0 and n_high > 0:
        weight_low = n_total / (3 * n_low)
        weight_medium = n_total / (3 * n_medium)
        weight_high = n_total / (3 * n_high)
    else:
        # Fallback to equal weights if any class is empty
        weight_low = weight_medium = weight_high = 1.0
    
    print(f"Class weights: Low={weight_low:.2f}, Medium={weight_medium:.2f}, High={weight_high:.2f}")
    
    return {
        'low_thresh': low_thresh,
        'high_thresh': high_thresh,
        'weight_low': weight_low,
        'weight_medium': weight_medium,
        'weight_high': weight_high
    }

class MemoryEfficientWeightedL1Loss(nn.Module):
    """Memory-efficient weighted L1 loss that computes weights per batch."""
    
    def __init__(self, weight_dict):
        super().__init__()
        self.low_thresh = weight_dict['low_thresh']
        self.high_thresh = weight_dict['high_thresh']
        self.weight_low = weight_dict['weight_low']
        self.weight_medium = weight_dict['weight_medium']
        self.weight_high = weight_dict['weight_high']
        
    def forward(self, pred, target):
        # Compute L1 loss
        l1_loss = torch.abs(pred - target)
        
        # Compute weights for this batch only (memory efficient)
        weights = torch.ones_like(target, dtype=torch.float32)
        low_mask = target <= self.low_thresh
        medium_mask = (target > self.low_thresh) & (target <= self.high_thresh)
        high_mask = target > self.high_thresh
        
        weights[low_mask] = self.weight_low
        weights[medium_mask] = self.weight_medium
        weights[high_mask] = self.weight_high
        
        # Apply weights
        weighted_loss = l1_loss * weights
        return weighted_loss.mean()

# ───────────────────────────── splits & loaders ---------------------------------

def temporal_split(X, y, timestamps, batch, splits=(0.7, 0.15, 0.15)):
    """
    Create chronological train/val/test splits for time series forecasting.
    
    Args:
        X: encoded features
        y: target values
        timestamps: array of timestamps corresponding to each sample
        batch: batch size
        splits: (train_pct, val_pct, test_pct) - should sum to 1.0
    
    Returns:
        train_loader, val_loader, test_loader, test_win_idx
    """
    # Sort by timestamp to ensure chronological order
    sorted_indices = np.argsort(timestamps)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    timestamps_sorted = timestamps[sorted_indices]
    
    # Calculate split points based on timestamps
    n = len(X_sorted)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    
    # Split indices
    i_train = sorted_indices[:n_train]
    i_val = sorted_indices[n_train:n_train + n_val]
    i_test = sorted_indices[n_train + n_val:]
    
    print(f"Chronological split:")
    print(f"  Train: {len(i_train):,} samples ({splits[0]*100:.0f}%) - {timestamps_sorted[0]} to {timestamps_sorted[n_train-1]}")
    print(f"  Val:   {len(i_val):,} samples ({splits[1]*100:.0f}%) - {timestamps_sorted[n_train]} to {timestamps_sorted[n_train+n_val-1]}")
    print(f"  Test:  {len(i_test):,} samples ({splits[2]*100:.0f}%) - {timestamps_sorted[n_train+n_val]} to {timestamps_sorted[-1]}")

    def to_loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(X[ix]), torch.from_numpy(y[ix]))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    return (
        to_loader(i_train, True),
        to_loader(i_val, False),
        to_loader(i_test, False),
        i_test,
    )

# ───────────────────────────── train / eval ------------------------------------

def epoch_loop(model, loader, optim=None, weighted_loss_fn=None):
    """
    Training/evaluation loop with optional weighted loss.
    
    Args:
        model: LSTM model
        loader: DataLoader
        optim: optimizer (None for evaluation)
        weighted_loss_fn: MemoryEfficientWeightedL1Loss instance (None for standard L1Loss)
    """
    if weighted_loss_fn is None:
        loss_fn = nn.L1Loss()
    else:
        loss_fn = weighted_loss_fn
        
    train_mode = optim is not None
    model.train(train_mode)
    total, n = 0.0, 0
    
    for xb, yb in loader:
        xb = xb.to(DEVICE, dtype=torch.float32)
        yb = yb.to(DEVICE, dtype=torch.float32).squeeze()
        
        if train_mode:
            optim.zero_grad()
            
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        total += loss.item() * len(xb)
        n += len(xb)
        
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
    return total / n

# ───────────────────────────── main --------------------------------------------

def main(args):
    print("Loading data...")
    df = pd.read_csv(args.csv)
    df['sensor_id'] = df['Latitude'].round(6).astype(str) + ';' + df['Longitude'].round(6).astype(str)
    df = df.sort_values(['sensor_id', 'Time']).reset_index(drop=True)

    print("Encoding data...")
    enc = TrafficDataEncoder(seq_len=args.seq, horizon=args.h).fit(df)
    X, y = enc.transform(df)
    
    # Extract timestamps for the encoded samples
    # The encoder creates windows, so we need to get timestamps for the target rows
    target_row_indices = np.arange(args.seq + args.h - 1, len(df))
    timestamps = df.iloc[target_row_indices]['Time'].values
    
    print(f"Encoded {len(X)} samples with timestamps from {timestamps[0]} to {timestamps[-1]}")
    
    # Clear some memory after encoding
    del df
    import gc
    gc.collect()

    # Compute class weights if using weighted loss
    weighted_loss_fn = None
    if args.use_weighted_loss:
        print("Computing class weights for weighted loss...")
        weight_dict = compute_speed_class_weights(y, speed_thresholds=(35, 55))
        weighted_loss_fn = MemoryEfficientWeightedL1Loss(weight_dict)
        print("Using memory-efficient weighted loss to handle class imbalance")

    print("Creating data loaders...")
    train_loader, val_loader, test_loader, test_win_idx = temporal_split(X, y, timestamps, args.batch)

    print("Initializing model...")
    model = LSTMReg(n_feats=X.shape[2], hidden=args.hidden).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    best_val = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = epoch_loop(model, train_loader, optim, weighted_loss_fn)
        val = epoch_loop(model, val_loader)  # Always use standard loss for validation
        print(f"Epoch {epoch:02d}  train MAE {tr:.3f}  val MAE {val:.3f}")
        if val < best_val:
            best_val, best_state = val, model.state_dict()

    model.load_state_dict(best_state) # type: ignore
    test_mae = epoch_loop(model, test_loader)  # Always use standard loss for testing
    print(f"Test MAE: {test_mae:.3f}")

    # save best model if requested
    if args.model_out:
        torch.save(best_state, args.model_out)
        print("Saved best model →", args.model_out)

    # ── Export predictions with meta columns ──────────────────────────────────
    if args.pred_csv:
        print("Generating predictions...")
        model.eval()
        preds = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                preds.append(model(xb).squeeze().cpu().numpy())
        preds = np.concatenate(preds)
        trues = y[test_win_idx].squeeze()

        # Reload data for metadata (only the rows we need)
        print("Loading metadata for predictions...")
        df_meta = pd.read_csv(args.csv)
        df_meta['sensor_id'] = df_meta['Latitude'].round(6).astype(str) + ';' + df_meta['Longitude'].round(6).astype(str)
        df_meta = df_meta.sort_values(['sensor_id', 'Time']).reset_index(drop=True)
        
        tgt_row_idx = test_win_idx + args.seq + args.h - 1
        meta_cols = ['Latitude', 'Longitude', 'direction', 'Time', 'AggSpeed']
        meta_df = df_meta.iloc[tgt_row_idx][meta_cols].reset_index(drop=True)
        meta_df['y_true'] = trues
        meta_df['y_pred'] = preds
        meta_df.to_csv(args.pred_csv, index=False)
        print(f"Saved test predictions → {args.pred_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=256, help="batch size (use smaller values like 128 or 256 for memory-constrained systems)")
    p.add_argument("--seq", type=int, default=12)
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pred_csv", type=str, default=None,
                   help="write preds with geo/meta columns to this CSV")
    p.add_argument("--model_out", type=str, default=None,
                   help="file to save best model state_dict")
    p.add_argument("--use_weighted_loss", action="store_true",
                   help="use weighted loss to handle class imbalance in speed data")
    args = p.parse_args()

    main(args)


"""
# 1. Create balanced dataset (one-time)
python model/resample_data.py --input data_process/exmaple.csv --output data_process/balanced_example.csv

# 2. Train on balanced data (no weights needed)
python model/train_lstm.py --csv data_process/balanced_example.csv \
       --epochs 50 --batch 256 --model_out lstm_balanced.pt \
       --pred_csv test_preds_balanced.csv

# 3. Evaluate
python model/evalute.py --csv data_process/exmaple.csv \
       --encoder encoder.pkl --model lstm_balanced.pt \
       --out final_predictions.csv
"""