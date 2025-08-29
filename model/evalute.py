# evaluate.py – run the saved lstm.pt on the hold-out part of example.csv
from pathlib import Path
import argparse, joblib, torch, numpy as np, pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from encode import TrafficDataEncoder   # your existing encoder class

# ── same architecture you trained ───────────────────────────────────────────
class LSTMReg(nn.Module):
    def __init__(self, n_feats, hidden=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feats, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2),
                                  nn.ReLU(), nn.Linear(hidden//2, 1))
    def forward(self, x):                 # x (B,T,F)
        return self.head(self.lstm(x)[0][:, -1])  # last step only

# ── helper ------------------------------------------------------------------
def MAE(pred, true): return np.abs(pred-true).mean()

# ── main --------------------------------------------------------------------
def main(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on", device)

    # ---------- data --------------------------------------------------------
    df = pd.read_csv(args.csv)
    df["sensor_id"] = (df["Latitude"].round(6).astype(str) + ";" +
                       df["Longitude"].round(6).astype(str))
    df = df.sort_values(["sensor_id", "Time"]).reset_index(drop=True)

    enc: TrafficDataEncoder = joblib.load(args.encoder)
    X, y = enc.transform(df)                         # (N_win, seq, F)

    # Extract timestamps for the encoded samples (same logic as training)
    seq_len = 12  # default from training
    horizon = 1   # default from training
    target_row_indices = np.arange(seq_len + horizon - 1, len(df))
    timestamps = df.iloc[target_row_indices]['Time'].values
    
    # Sort by timestamp to ensure chronological order
    sorted_indices = np.argsort(timestamps)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # replicate the 70/15/15 time split used in training
    n = len(X_sorted); n_train = int(n*0.70); n_val = int(n*0.15)
    X_test, y_test = X_sorted[n_train+n_val:], y_sorted[n_train+n_val:]

    # ---------- model -------------------------------------------------------
    net = LSTMReg(n_feats=X.shape[2], hidden=args.hidden).to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    dl = DataLoader(TensorDataset(torch.from_numpy(X_test),
                                  torch.from_numpy(y_test)),
                    batch_size=1024)

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, dtype=torch.float32)
            preds.append(net(xb).squeeze().cpu().numpy())
            trues.append(yb.squeeze().numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)

    print(f"Test MAE: {MAE(preds, trues):.3f} mph")

    if args.out:
        pd.DataFrame({"y_true": trues, "y_pred": preds}) \
          .to_csv(args.out, index=False)
        print("Saved raw predictions →", args.out)

# ---------- CLI -------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",      required=True)
    ap.add_argument("--encoder",  required=True, help="encoder.pkl")
    ap.add_argument("--model",    required=True, help="lstm.pt (state_dict)")
    ap.add_argument("--hidden",   type=int, default=64)
    ap.add_argument("--out",      default=None)
    main(ap.parse_args())


"""
python model/evaluate.py \
       --csv /Users/amitomer/Desktop/Personal/University/deep_learning/TrafCast/data_process/exmaple.csv \
       --encoder encoder.pkl \
       --model lstm.pt \
       --out raw_test_preds.csv

"""