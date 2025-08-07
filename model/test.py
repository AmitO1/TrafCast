# test.py – predict **one specific timestamp** for all sensors using saved full model
# ---------------------------------------------------------------------------
# Requirements:
#   * You saved the entire Module (torch.save(model, 'full_lstm.pt'))
#   * You saved the fitted encoder (encoder.pkl)
#
# Example:
#   python model/test.py \
#          --csv data_process/example.csv \
#          --encoder encoder.pkl \
#          --model full_lstm.pt \
#          --timestamp "2025-03-01 03:00:00" \
#          --out preds_at_0300.csv
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import joblib

from encode import TrafficDataEncoder

# ───────────────────────────── helper -----------------------------------------

def pick_windows_at_timestamp(df: pd.DataFrame, enc: TrafficDataEncoder, ts: str):
    """Return X_sel (n_sensors, seq_len, F) and row indices with Time == ts."""
    seq_len, horizon = enc.seq_len, enc.horizon
    # full transform so we can slice
    X_all, _ = enc.transform(df)
    start_row = seq_len + horizon - 1  # first target row represented in X_all

    target_mask = df['Time'] == ts
    if not target_mask.any():
        print(f"Timestamp {ts} not found in DataFrame", file=sys.stderr)
        sys.exit(1)

    tgt_rows = np.where(target_mask)[0]
    win_idxs = tgt_rows - start_row
    # filter rows that are too early (< 0) or came from gaps
    valid = win_idxs >= 0
    win_idxs = win_idxs[valid]
    tgt_rows = tgt_rows[valid]
    X_sel = X_all[win_idxs]
    return X_sel, tgt_rows

# ───────────────────────────── main -------------------------------------------

def main(args):
    # device setup
    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print("Predicting on", device)

    # load & sort data
    df = pd.read_csv(args.csv)
    df['sensor_key'] = df['Latitude'].round(6).astype(str) + ';' + df['Longitude'].round(6).astype(str)
    df = df.sort_values(['sensor_key', 'Time']).reset_index(drop=True)

    # load encoder + model
    enc: TrafficDataEncoder = joblib.load(args.encoder)
    model = torch.load(args.model, map_location=device)
    model.to(device).eval()

    # build windows for given timestamp
    X_sel, tgt_rows = pick_windows_at_timestamp(df, enc, args.timestamp)
    if len(X_sel) == 0:
        print("No valid windows (not enough history) for timestamp", args.timestamp)
        sys.exit(1)

    with torch.no_grad():
        preds = model(torch.from_numpy(X_sel).to(device, dtype=torch.float32)).squeeze().cpu().numpy()

    meta_cols = ['Latitude', 'Longitude', 'direction', 'road_name', 'Time', 'AggSpeed']
    out_df = df.iloc[tgt_rows][meta_cols].reset_index(drop=True)
    out_df['y_pred'] = preds
    out_df.to_csv(args.out, index=False)
    print(f"Saved {len(out_df)} predictions → {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--timestamp", required=True, help="YYYY-MM-DD HH:MM:SS of desired prediction")
    p.add_argument("--out", default="preds_at_ts.csv")
    args = p.parse_args()
    main(args)
