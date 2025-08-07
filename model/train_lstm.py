# train_lstm.py – save best model + device support + rich export
# ---------------------------------------------------------------------------
# Adds --model_out option to store the best-performing state_dict.
# Example:
#   python model/train_lstm.py --csv data_process/example.csv \
#          --epochs 8 --batch 512 --model_out lstm.pt --pred_csv test_preds.csv
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

# ───────────────────────────── splits & loaders ---------------------------------

def temporal_split(X, y, batch, splits=(0.7, 0.15, 0.15)):
    n = len(X)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    idxs = np.arange(n)
    i_train, i_val, i_test = np.split(idxs, [n_train, n_train + n_val])

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

def epoch_loop(model, loader, optim=None):
    loss_fn = nn.L1Loss()
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
    df = pd.read_csv(args.csv)
    df['sensor_id'] = df['Latitude'].round(6).astype(str) + ';' + df['Longitude'].round(6).astype(str)
    df = df.sort_values(['sensor_id', 'Time']).reset_index(drop=True)

    enc = TrafficDataEncoder(seq_len=args.seq, horizon=args.h).fit(df)
    X, y = enc.transform(df)

    train_loader, val_loader, test_loader, test_win_idx = temporal_split(X, y, args.batch)

    model = LSTMReg(n_feats=X.shape[2], hidden=args.hidden).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        tr = epoch_loop(model, train_loader, optim)
        val = epoch_loop(model, val_loader)
        print(f"Epoch {epoch:02d}  train MAE {tr:.3f}  val MAE {val:.3f}")
        if val < best_val:
            best_val, best_state = val, model.state_dict()

    model.load_state_dict(best_state) # type: ignore
    test_mae = epoch_loop(model, test_loader)
    print(f"Test MAE: {test_mae:.3f}")

    # save best model if requested
    if args.model_out:
        torch.save(best_state, args.model_out)
        print("Saved best model →", args.model_out)

    # ── Export predictions with meta columns ──────────────────────────────────
    if args.pred_csv:
        model.eval()
        preds = []
        for xb, _ in test_loader:
            xb = xb.to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                preds.append(model(xb).squeeze().cpu().numpy())
        preds = np.concatenate(preds)
        trues = y[test_win_idx].squeeze()

        tgt_row_idx = test_win_idx + args.seq + args.h - 1
        meta_cols = ['Latitude', 'Longitude', 'direction', 'road_name', 'Time', 'AggSpeed']
        meta_df = df.iloc[tgt_row_idx][meta_cols].reset_index(drop=True)
        meta_df['y_true'] = trues
        meta_df['y_pred'] = preds
        meta_df.to_csv(args.pred_csv, index=False)
        print(f"Saved test predictions → {args.pred_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seq", type=int, default=12)
    p.add_argument("--h", type=int, default=1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--pred_csv", type=str, default=None,
                   help="write preds with geo/meta columns to this CSV")
    p.add_argument("--model_out", type=str, default=None,
                   help="file to save best model state_dict")
    args = p.parse_args()

    main(args)
