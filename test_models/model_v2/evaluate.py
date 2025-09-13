# model/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

from encode import TrafficDataEncoder

# ───────────────────────── device ─────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# ───────────────────────── model ──────────────────────────
class LSTMReg(nn.Module):
    def __init__(self, n_feats: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # last timestep

# ─────────────────────── helpers ─────────────────────────
def _parse_targets(args) -> List[pd.Timestamp]:
    if args.at:
        return [pd.to_datetime(args.at)]
    if args.from_dt and args.to_dt:
        freq = args.freq or "5min"
        rng = pd.date_range(pd.to_datetime(args.from_dt), pd.to_datetime(args.to_dt), freq=freq, inclusive="both")
        return list(rng.to_pydatetime())
    raise ValueError("Provide either --at <timestamp> OR --from_dt/--to_dt [--freq].")

def _ensure_sensor_id(df: pd.DataFrame) -> pd.DataFrame:
    if "sensor_id" not in df.columns:
        df["sensor_id"] = df["Latitude"].round(6).astype(str) + ";" + df["Longitude"].round(6).astype(str)
    return df

def _load_encoder_or_fit(df: pd.DataFrame, args, fit_cutoff: pd.Timestamp) -> TrafficDataEncoder:
    if args.encoder:
        print(f"Loading encoder from {args.encoder}")
        enc: TrafficDataEncoder = joblib.load(args.encoder)  # type: ignore
        return enc
    df_fit = df[df["Time"] < fit_cutoff].copy()
    if len(df_fit) < 1000:
        print("Warning: too few rows before target to fit encoder; fitting on full data (may leak).")
        df_fit = df
    enc = TrafficDataEncoder(
        seq_len=args.seq, horizon=args.h,
        stride=1, max_windows_per_sensor=None,
        keep_prob_low=1.0, keep_prob_med=1.0, keep_prob_high=1.0, rng_seed=42
    ).fit(df_fit)
    print("Fitted encoder on", len(df_fit), "rows (cutoff:", fit_cutoff, ")")
    return enc

def _build_group_features(df_proc: pd.DataFrame, enc: TrafficDataEncoder, target_col: str):
    """Return per-sensor cached arrays for fast window slicing, including y_true."""
    cat_arr = enc._ordinal_encoder.transform(df_proc[enc._cat_cols]).astype(np.float32)  # type: ignore
    num_arr = enc._scaler.transform(df_proc[enc._num_cols]).astype(np.float32)           # type: ignore
    feats_all = np.concatenate([num_arr, cat_arr], axis=1)
    y_all = df_proc[target_col].to_numpy(dtype=np.float32, copy=False)

    group_cache = {}
    for sid, g in df_proc.groupby("sensor_id", sort=False):
        g_idx = g.index.to_numpy()
        group_cache[sid] = {
            "index": g_idx,
            "times": g["Time"].to_numpy(),
            "feats": feats_all[g_idx],
            "y_true": y_all[g_idx],
            "meta": g[["Latitude", "Longitude"]].iloc[-1],
        }
    return group_cache

# ─────────────────────── main predict ────────────────────
def main(args):
    # Load & basic prep
    df = pd.read_csv(args.csv)
    df = _ensure_sensor_id(df)
    road_col = args.road_col if args.road_col and args.road_col in df.columns else None
    if args.road and road_col:
        df = df[df[road_col] == args.road].copy()
        if df.empty:
            raise ValueError(f"No rows for road '{args.road}' in column '{road_col}'.")
        print(f"Filtered road '{args.road}' → {len(df):,} rows")
    elif args.road and not road_col:
        print("Warning: --road given but --road_col missing or not in CSV; ignoring road filter.")

    # Parse time & sort
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values(["sensor_id", "Time"]).reset_index(drop=True)

    # Targets
    targets = _parse_targets(args)
    targets = sorted(pd.to_datetime(t) for t in targets)
    first_t = targets[0]
    print(f"Targets: {targets[0]} ... {targets[-1]} ({len(targets)} total)")

    # Encoder
    enc = _load_encoder_or_fit(df, args, fit_cutoff=first_t)

    # Preprocess to match training path
    df_proc = enc._preprocess(df)

    # Build per-sensor caches
    group_cache = _build_group_features(df_proc, enc, target_col=args.target_col)

    # Alignment (5-min granularity assumed)
    step_minutes = 5
    offset = pd.Timedelta(minutes=step_minutes * enc.horizon)
    seq = enc.seq_len

    windows: List[np.ndarray] = []
    rows_meta: List[Tuple[str, float, float, pd.Timestamp, float]] = []  # (sid, lat, lon, target_time, y_true)

    for sid, blob in group_cache.items():
        times = blob["times"]
        feats = blob["feats"]
        y_true_series = blob["y_true"]
        lat, lon = float(blob["meta"]["Latitude"]), float(blob["meta"]["Longitude"])

        for t in targets:
            target_time = pd.Timestamp(t)
            context_end_time = target_time - offset
            idx = np.searchsorted(times, np.datetime64(context_end_time))
            if idx == len(times) or times[idx] != np.datetime64(context_end_time):
                continue
            start = idx - (seq - 1)
            if start < 0:
                continue
            tgt_row = idx + enc.horizon
            if tgt_row >= len(times):
                continue

            win = feats[start : idx + 1]
            if win.shape[0] != seq:
                continue

            # y_true at the target row (may be NaN if missing in CSV)
            y_true_val = float(y_true_series[tgt_row]) if not np.isnan(y_true_series[tgt_row]) else np.nan

            windows.append(win)
            rows_meta.append((sid, lat, lon, target_time, y_true_val))

    if not windows:
        raise RuntimeError("No valid windows could be formed for the requested time(s). "
                           "Check that your CSV has 5-minute samples and enough history before the target time.")

    X = np.stack(windows, axis=0)  # (N, seq, F)
    print(f"Prepared {X.shape[0]} windows for prediction.")

    # Load model & predict
    n_feats = X.shape[2]
    model = LSTMReg(n_feats=n_feats, hidden=args.hidden).to(DEVICE)
    state = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds = []
    bs = args.batch
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).to(DEVICE, dtype=torch.float32)
            yb = model(xb).squeeze().detach().cpu().numpy()
            preds.append(yb)
    y_pred = np.concatenate(preds)

    # Build output dataframe (now with y_true)
    out = pd.DataFrame({
        "road": (args.road if args.road else (df[road_col].iloc[0] if road_col else None)),
        "sensor_id": [r[0] for r in rows_meta],
        "Latitude": [r[1] for r in rows_meta],
        "Longitude": [r[2] for r in rows_meta],
        "target_time": [r[3] for r in rows_meta],
        "y_true": [r[4] for r in rows_meta],
        "y_pred": y_pred,
    }).sort_values(["target_time", "sensor_id"]).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved predictions → {args.out}")
    print(out.head(5).to_string(index=False))

# ──────────────────────── entrypoint ─────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate LSTM model at a specific time (or range) per road.")
    ap.add_argument("--csv", required=True, help="Raw data CSV (same schema as training).")
    ap.add_argument("--model", required=True, help="Path to model state_dict (.pt) from training.")
    ap.add_argument("--encoder", default=None, help="Path to fitted encoder .pkl (joblib). If absent, fit on rows before first target.")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV path.")
    ap.add_argument("--road_col", type=str, default="ref", help="Column name for road id (e.g., 'ref').")
    ap.add_argument("--road", type=str, default=None, help="Road value to filter (e.g., 'US-101'). If omitted, uses all roads.")
    ap.add_argument("--seq", type=int, default=12, help="Sequence length (must match training).")
    ap.add_argument("--h", type=int, default=1, help="Horizon (must match training).")
    ap.add_argument("--hidden", type=int, default=128, help="Hidden size used at training (to rebuild model).")
    ap.add_argument("--batch", type=int, default=1024, help="Batch size for inference.")
    ap.add_argument("--target_col", type=str, default="AggSpeed", help="Name of target column in CSV.")
    # choose exactly one of these:
    ap.add_argument("--at", type=str, default=None, help="Single timestamp, e.g. '2025-03-28 08:30'")
    ap.add_argument("--from_dt", type=str, default=None, help="Start datetime for a range")
    ap.add_argument("--to_dt", type=str, default=None, help="End datetime for a range")
    ap.add_argument("--freq", type=str, default=None, help="Frequency for range (default 5min)")

    args = ap.parse_args()
    main(args)


"""
python model_v2/evaluate.py --csv /Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/data_process/example.csv \
  --model lstm_huber.pt --encoder encoder.pkl \
  --road_col ref --road "I 405" \
  --seq 12 --h 1 \
  --at "2025-03-28 08:30" \
  --out preds_101_2025-03-28_0830.csv
"""