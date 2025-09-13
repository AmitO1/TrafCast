# evaluate.py – predict for ALL sensors at a given timestamp
from pathlib import Path
import argparse, joblib, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from encode import TrafficDataEncoder   # your existing encoder class


# ── model (same architecture as training) ────────────────────────────────────
class LSTMReg(nn.Module):
    def __init__(self, n_feats, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):                # x: (B, T, F)
        # last step only
        return self.head(self.lstm(x)[0][:, -1])


def _pick_seq_len_horizon_from_encoder(enc, seq_len_arg, horizon_arg):
    """
    Best-effort to read seq_len/horizon from the fitted encoder; fall back to CLI.
    """
    seq_len = getattr(enc, "seq_len", None)
    horizon = getattr(enc, "horizon", None)
    if seq_len is None:
        seq_len = seq_len_arg
    if horizon is None:
        horizon = horizon_arg
    return int(seq_len), int(horizon)


def main(args):
    # ── device ───────────────────────────────────────────────────────────────
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)

    # ── load and sort raw data ───────────────────────────────────────────────
    df = pd.read_csv(args.csv)

    # Ensure we have a unique sensor identifier (you already used this in older eval)
    if "sensor_id" not in df.columns:
        df["sensor_id"] = (df["Latitude"].round(6).astype(str) + ";" +
                           df["Longitude"].round(6).astype(str))

    # Parse Time to datetime for robust equality (and sort)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["sensor_id", "Time"]).reset_index(drop=True)

    # ── load encoder and transform ───────────────────────────────────────────
    enc: TrafficDataEncoder = joblib.load(args.encoder)
    seq_len, horizon = _pick_seq_len_horizon_from_encoder(enc, args.seq_len, args.horizon)

    # X: (N, seq_len, F), y: (N, 1) or (N,) depending on your encoder
    X, y = enc.transform(df)

    # ── compute the target row index in df for each encoded sample ───────────
    # For a single continuous timeseries, target_row_indices = np.arange(seq_len+horizon-1, len(df)).
    # But since we have multiple sensors concatenated, we must compute this per sensor to avoid
    # crossing boundaries. We'll do it by grouping and assembling indices.
    target_row_indices = []
    for _, g in df.groupby("sensor_id", sort=False):
        n = len(g)
        start = g.index.min()
        # valid targets within this sensor segment:
        local_targets = np.arange(seq_len + horizon - 1, n)
        # map to global row indices
        target_row_indices.extend(list(start + local_targets))
    target_row_indices = np.array(target_row_indices, dtype=int)

    if len(target_row_indices) != len(X):
        # Safety check: if your encoder already avoids crossing sensors, these should match.
        # If not, we try the simpler global fallback (older training logic).
        print("[warn] target_row_indices length mismatch; using global fallback mapping.")
        target_row_indices = np.arange(seq_len + horizon - 1, seq_len + horizon - 1 + len(X))

    # ── build metadata for each encoded sample (timestamp & sensor info) ─────
    target_times = df.loc[target_row_indices, "Time"].values
    target_sensor_ids = df.loc[target_row_indices, "sensor_id"].values
    target_lat = df.loc[target_row_indices, "Latitude"].values
    target_lon = df.loc[target_row_indices, "Longitude"].values

    # ── select samples exactly at the requested time (with optional tolerance) ─
    target_dt = pd.to_datetime(args.when)
    if args.tolerance_mins > 0:
        delta = pd.Timedelta(minutes=args.tolerance_mins)
        mask = (target_times >= (target_dt - delta)) & (target_times <= (target_dt + delta))
    else:
        mask = (target_times == target_dt)

    if not mask.any():
        print(f"No encoded samples match time={target_dt} (tolerance={args.tolerance_mins} min).")
        print("Tip: increase --tolerance-mins if your raw timestamps aren’t perfectly aligned.")
        return

    # Subset to the requested timestamp
    X_sel = X[mask]
    y_sel = y[mask]

    # Convert y_sel to columns for the CSV
    def _ycols(y_block):
        y_block = np.asarray(y_block)
        if y_block.ndim == 1:            # horizon == 1
            return {"y_true": y_block}
        else:                            # horizon > 1 → y_true_t+1, y_true_t+2, ...
            cols = {}
            H = y_block.shape[1]
            for j in range(H):
                cols[f"y_true_t+{j+1}"] = y_block[:, j]
            return cols
            
    y_true_cols = _ycols(y_sel)

    times_sel = target_times[mask]
    sensor_sel = target_sensor_ids[mask]
    lat_sel = target_lat[mask]
    lon_sel = target_lon[mask]

    # ── model ────────────────────────────────────────────────────────────────
    net = LSTMReg(n_feats=X.shape[2], hidden=args.hidden).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    dl = DataLoader(TensorDataset(torch.from_numpy(X_sel)),
                    batch_size=args.batch_size)

    preds = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device, dtype=torch.float32)
            preds.append(net(xb).squeeze().cpu().numpy())
    preds = np.concatenate(preds)

    # ── assemble output ──────────────────────────────────────────────────────
    # If multiple rows per sensor landed in the tolerance window, keep them all.
    out = pd.DataFrame({
        "Time": pd.to_datetime(times_sel),
        "sensor_id": sensor_sel,
        "Latitude": lat_sel,
        "Longitude": lon_sel,
        "y_pred": preds,
        **y_true_cols,
    }).sort_values(["sensor_id", "Time"]).reset_index(drop=True)

    # Optional: if you want strictly one row per sensor, nearest to target_dt:
    if args.one_per_sensor:
        out["abs_dt_diff"] = (out["Time"] - target_dt).abs()
        out = out.sort_values(["sensor_id", "abs_dt_diff"]).groupby("sensor_id", as_index=False).first()
        out = out.drop(columns=["abs_dt_diff"]).sort_values("sensor_id")

    # Save
    out_path = args.out or f"preds_{target_dt.strftime('%Y%m%d_%H%M%S')}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} predictions → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict for all sensors at a given timestamp.")
    ap.add_argument("--csv",      required=True, help="Path to raw CSV (example.csv)")
    ap.add_argument("--encoder",  required=True, help="Path to fitted encoder.pkl")
    ap.add_argument("--model",    required=True, help="Path to lstm.pt (state_dict)")
    ap.add_argument("--when",     required=True, help="Target timestamp, e.g. '2025-03-15 17:45'")
    ap.add_argument("--out",      default=None,  help="Output CSV path")
    ap.add_argument("--hidden",   type=int, default=128, help="Hidden size used in training")
    ap.add_argument("--seq-len",  type=int, default=12, help="Fallback seq_len if encoder lacks it")
    ap.add_argument("--horizon",  type=int, default=1,  help="Fallback horizon if encoder lacks it")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tolerance-mins", type=int, default=0,
                    help="Allow ±N minutes matching around --when (0 = exact match)")
    ap.add_argument("--one-per-sensor", action="store_true",
                    help="If set, keep only the row closest to --when for each sensor")
    main(ap.parse_args())

