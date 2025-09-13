# train_lstm.py – save best model + device support + rich export + memory-efficient weighted loss
# ----------------------------------------------------------------------------------------------
# Usage example:
# !python model/train_lstm.py --csv data_process/example.csv \
#   --epochs 5 --batch 128 --hidden 128 --loss l1 --model_out lstm_balanced.pt \
#   --pred_csv test_preds_balanced.csv --use_weighted_loss \
#   --road_col ref --road_frac 0.5 --sensor_frac_per_road 0.5 \
#   --stride 3 --max_windows_per_sensor 4000 \
#   --keep_prob_low 1.0 --keep_prob_med 0.8 --keep_prob_high 0.25
# ----------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
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
    def __init__(self, n_feats: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.3):
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
    y: (N, horizon) or (N,)
    """
    low_thresh, high_thresh = speed_thresholds
    y1 = y[:, -1] if (y.ndim == 2 and y.shape[1] >= 1) else y

    low_speed = y1 <= low_thresh
    med_speed = (y1 > low_thresh) & (y1 <= high_thresh)
    high_speed = y1 > high_thresh

    n_low = int(low_speed.sum())
    n_med = int(med_speed.sum())
    n_high = int(high_speed.sum())
    n_total = len(y1)

    print("Speed distribution:")
    print(f"  Low (≤{low_thresh}): {n_low} ({n_low / max(1,n_total) * 100:.1f}%)")
    print(f"  Med ({low_thresh}-{high_thresh}]: {n_med} ({n_med / max(1,n_total) * 100:.1f}%)")
    print(f"  High (>{high_thresh}): {n_high} ({n_high / max(1,n_total) * 100:.1f}%)")

    if n_low > 0 and n_med > 0 and n_high > 0:
        w_low = n_total / (3 * n_low)
        w_med = n_total / (3 * n_med)
        w_high = n_total / (3 * n_high)
    else:
        w_low = w_med = w_high = 1.0
    
    w_low *= args.boost_low
    w_med *= args.boost_med

    print(f"Class weights: Low={w_low:.2f}, Med={w_med:.2f}, High={w_high:.2f}")
    return {
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
        "weight_low": w_low,
        "weight_medium": w_med,
        "weight_high": w_high,
    }

class MemoryEfficientWeightedL1Loss(nn.Module):
    """Memory-efficient weighted L1 loss that computes weights per batch."""
    def __init__(self, weight_dict):
        super().__init__()
        self.low_thresh = weight_dict["low_thresh"]
        self.high_thresh = weight_dict["high_thresh"]
        self.weight_low = weight_dict["weight_low"]
        self.weight_medium = weight_dict["weight_medium"]
        self.weight_high = weight_dict["weight_high"]

    def forward(self, pred, target):
        if target.ndim > 1:  # (N, horizon)
            target = target[:, -1]
        l1 = torch.abs(pred - target)
        weights = torch.ones_like(target, dtype=torch.float32)
        low_mask = target <= self.low_thresh
        med_mask = (target > self.low_thresh) & (target <= self.high_thresh)
        high_mask = target > self.high_thresh
        weights[low_mask] = self.weight_low
        weights[med_mask] = self.weight_medium
        weights[high_mask] = self.weight_high
        return (l1 * weights).mean()

# ───────────────────────────── splits & loaders --------------------------------
def temporal_split(X, y, timestamps, batch, splits=(0.7, 0.15, 0.15)):
    """
    Chronological train/val/test split.
    Returns train_loader, val_loader, test_loader, test_indices
    """
    ts = pd.to_datetime(pd.Series(timestamps))
    is_nat = ts.isna().to_numpy()
    valid_idx = np.where(~is_nat)[0]
    nat_idx = np.where(is_nat)[0]
    order_valid = valid_idx[np.argsort(ts.iloc[valid_idx].to_numpy())]
    sorted_indices = np.concatenate([order_valid, nat_idx], axis=0)

    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    ts_sorted = ts.iloc[sorted_indices].to_numpy()

    n = len(X_sorted)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    i_train = sorted_indices[:n_train]
    i_val = sorted_indices[n_train:n_train + n_val]
    i_test = sorted_indices[n_train + n_val:]

    print("Chronological split:")
    print(f"  Train: {len(i_train):,} ({splits[0]*100:.0f}%) - {ts_sorted[0]} → {ts_sorted[n_train-1] if n_train>0 else 'NA'}")
    print(f"  Val:   {len(i_val):,} ({splits[1]*100:.0f}%) - {ts_sorted[n_train] if n_train<n else 'NA'} → {ts_sorted[n_train+n_val-1] if (n_train+n_val)>0 else 'NA'}")
    print(f"  Test:  {len(i_test):,} ({splits[2]*100:.0f}%) - {ts_sorted[n_train+n_val] if (n_train+n_val)<n else 'NA'} → {ts_sorted[-1] if n>0 else 'NA'}")

    def to_loader(ix, shuffle):
        ds = TensorDataset(torch.from_numpy(X[ix]), torch.from_numpy(y[ix]))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    return to_loader(i_train, True), to_loader(i_val, False), to_loader(i_test, False), i_test

# ───────────────────────────── train / eval ------------------------------------
def epoch_loop(model, loader, optim=None, weighted_loss_fn=None):
    """Training/evaluation loop with optional weighted loss."""
    if weighted_loss_fn is None:
        loss_fn = nn.L1Loss() if args.loss == "l1" else nn.SmoothL1Loss(beta=args.huber_beta)
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

    return total / max(1, n)

# ───────────────────────────── main --------------------------------------------
def main(args):
    print("Loading data...")
    df = pd.read_csv(args.csv)

    # Build sensor_id early
    df["sensor_id"] = df["Latitude"].round(6).astype(str) + ";" + df["Longitude"].round(6).astype(str)

    # Subsample roads and sensors (to control dataset size)
    rng = np.random.default_rng(42)
    road_col = args.road_col if args.road_col and args.road_col in df.columns else None

    if road_col:
        all_roads = df[road_col].dropna().unique()
        k_roads = max(1, int(len(all_roads) * args.road_frac))
        keep_roads = set(rng.choice(all_roads, size=k_roads, replace=False))
        df = df[df[road_col].isin(keep_roads)].reset_index(drop=True)
        print(f"Kept {k_roads}/{len(all_roads)} roads via road_frac={args.road_frac}")

        keep_sensors = []
        for _, g in df.groupby(road_col, sort=False):
            sensors = g["sensor_id"].unique()
            k = max(1, int(len(sensors) * args.sensor_frac_per_road))
            keep_sensors.extend(rng.choice(sensors, size=k, replace=False))
        keep_sensors = set(keep_sensors)
        df = df[df["sensor_id"].isin(keep_sensors)].reset_index(drop=True)
        print(f"Per-road sensor subsample via sensor_frac_per_road={args.sensor_frac_per_road}")
    else:
        sensors = df["sensor_id"].unique()
        k = max(1, int(len(sensors) * args.sensor_frac_per_road))
        keep_sensors = set(rng.choice(sensors, size=k, replace=False))
        df = df[df["sensor_id"].isin(keep_sensors)].reset_index(drop=True)
        print(f"No road_col. Kept {k}/{len(sensors)} sensors globally.")

    # Sort for stable windowing
    df = df.sort_values(["sensor_id", "Time"]).reset_index(drop=True)

    print("Encoding data...")
    enc = TrafficDataEncoder(
        seq_len=args.seq,
        horizon=args.h,
        stride=args.stride,
        max_windows_per_sensor=args.max_windows_per_sensor,
        keep_prob_low=args.keep_prob_low,
        keep_prob_med=args.keep_prob_med,
        keep_prob_high=args.keep_prob_high,
        rng_seed=42,
    ).fit(df)

    import joblib
    if getattr(args, "encoder_out", None):
        joblib.dump(enc, args.encoder_out)
        print("Saved encoder →", args.encoder_out)

    # Encoder now returns (X, y, target_row_indices, tgt_times)
    X, y, target_row_indices, timestamps = enc.transform(df)
    print("After downsampling:", X.shape[0], "windows")

    # Sanity check
    assert len(target_row_indices) == len(X) == len(y), \
        f"Mapping mismatch: targets={len(target_row_indices)} X={len(X)} y={len(y)}"

    # Timestamp summary
    ts_series = pd.to_datetime(pd.Series(timestamps))
    print(f"Encoded {len(X):,} samples with timestamps from {ts_series.min()} to {ts_series.max()}")

    # Free raw df (we'll reload for metadata later)
    del df
    import gc; gc.collect()

    # Weighted loss (optional)
    weighted_loss_fn = None
    if args.use_weighted_loss:
        print("Computing class weights for weighted loss...")
        weight_dict = compute_speed_class_weights(y, speed_thresholds=(35, 55))
        weighted_loss_fn = MemoryEfficientWeightedL1Loss(weight_dict)
        print("Using memory-efficient weighted loss to handle class imbalance")

    print("Creating data loaders...")
    train_loader, val_loader, test_loader, test_idx = temporal_split(X, y, timestamps, args.batch)

    print("Initializing model...")
    model = LSTMReg(n_feats=X.shape[2], hidden=args.hidden).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training...")
    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = epoch_loop(model, train_loader, optim, weighted_loss_fn)
        val = epoch_loop(model, val_loader)  # standard loss for validation
        print(f"Epoch {epoch:02d}  train MAE {tr:.3f}  val MAE {val:.3f}")
        if val < best_val:
            best_val, best_state = val, model.state_dict()

    # Load best and evaluate on test
    model.load_state_dict(best_state)  # type: ignore
    test_mae = epoch_loop(model, test_loader)  # standard loss for testing
    print(f"Test MAE: {test_mae:.3f}")

    # Save model
    if args.model_out:
        torch.save(best_state, args.model_out)
        print("Saved best model →", args.model_out)

    # Export predictions with meta
    if args.pred_csv:
        print("Generating predictions...")
        model.eval()
        preds = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                preds.append(model(xb).squeeze().cpu().numpy())
        preds = np.concatenate(preds)
        trues = y[test_idx].squeeze()

        print("Loading metadata for predictions...")
        df_meta = pd.read_csv(args.csv)

        # Recreate the same ordering the encoder used for indices:
        # apply the same ensure+sort step, then index with target_row_indices
        df_meta_proc = TrafficDataEncoder._ensure_sensor_id_and_sort(df_meta)
        tgt_row_idx = np.asarray(target_row_indices)[test_idx]

        meta_cols = ["Latitude", "Longitude", "direction", "Time", "AggSpeed"]
        meta_df = df_meta_proc.iloc[tgt_row_idx][meta_cols].reset_index(drop=True)
        meta_df["y_true"] = trues
        meta_df["y_pred"] = preds
        meta_df.to_csv(args.pred_csv, index=False)
        print(f"Saved test predictions → {args.pred_csv}")

# ───────────────────────────── entrypoint --------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=256, help="batch size (use smaller values like 128 or 256 for memory-constrained systems)")
    p.add_argument("--seq", type=int, default=12)
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--loss", choices=["l1", "huber"], default="l1")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pred_csv", type=str, default=None, help="write preds with geo/meta columns to this CSV")
    p.add_argument("--model_out", type=str, default=None, help="file to save best model state_dict")
    p.add_argument("--use_weighted_loss", action="store_true", help="use weighted loss to handle class imbalance in speed data")

    # Dataset size control knobs
    p.add_argument("--road_col", type=str, default="ref", help="column name for road/highway id (e.g., 'ref'); set None if not available")
    p.add_argument("--road_frac", type=float, default=0.5, help="fraction of roads to keep")
    p.add_argument("--sensor_frac_per_road", type=float, default=0.5, help="fraction of sensors per kept road")
    p.add_argument("--stride", type=int, default=3, help="take every k-th window (3 ≈ 15-min step with seq_len=12)")
    p.add_argument("--max_windows_per_sensor", type=int, default=4000, help="cap windows per sensor after stride")
    p.add_argument("--keep_prob_low", type=float, default=1.0, help="downsampling keep prob for low speeds (≤35)")
    p.add_argument("--keep_prob_med", type=float, default=0.7, help="downsampling keep prob for medium (35–55]")
    p.add_argument("--keep_prob_high", type=float, default=0.25, help="downsampling keep prob for high (>55)")
    p.add_argument("--boost_low", type=float, default=1.0, help="extra multiplier on low-speed loss")
    p.add_argument("--boost_med", type=float, default=1.0, help="extra multiplier on medium-speed loss")
    p.add_argument("--huber_beta", type=float, default=1.0, help="SmoothL1 beta (delta)")
    p.add_argument("--encoder_out", type=str, default=None,
               help="path to save fitted encoder (.pkl)")


    args = p.parse_args()
    main(args)

"""
# A) Robust loss - Best model
!python train_lstm.py --csv example.csv --epochs 25 --batch 128 --hidden 128 \
  --loss huber --model_out lstm_huber.pt --pred_csv preds_huber.csv --use_weighted_loss \
  --road_col ref --road_frac 0.5 --sensor_frac_per_road 0.5 \
  --stride 3 --max_windows_per_sensor 4000 \
  --keep_prob_low 1.0 --keep_prob_med 0.8 --keep_prob_high 0.25

"""