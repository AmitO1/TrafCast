from __future__ import annotations
"""encode.py – robust feature engineering for 5‑minute traffic speed data.
This version *imputes missing values* so you never propagate NaNs.

Pipeline
--------
1.  Geographic projection  (lat, lon) → local x_km, y_km (EPSG:6423)
2.  Time features           sin/cos hour‑of‑day & day‑of‑week
3.  Numeric clean‑up        lanes → int, maxspeed → float mph
4.  **Missing‑value fill**  categorical "UNK", numeric median
5.  Encoding                ordinal + StandardScaler
6.  Sliding windows         → (N, seq_len, features), target (N, horizon)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    import pyproj
    _TRANSFORMER = pyproj.Transformer.from_crs("epsg:4326", "epsg:6423", always_xy=True)
except Exception:
    def _fallback_ll_to_xy(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = 6_371_000  # metres
        x = np.deg2rad(lon) * R * np.cos(np.deg2rad(lat.mean()))
        y = np.deg2rad(lat) * R
        return x, y
    _TRANSFORMER = None

__all__ = ["TrafficDataEncoder"]


@dataclass
class TrafficDataEncoder:
    horizon: int = 1        # predict 1 × 5‑min step ahead
    seq_len: int = 12       # history length (12 × 5‑min = 1 h)

    _cat_cols: List[str] = field(default_factory=lambda: ["direction", "road_name", "weather"])
    _num_cols: List[str] = field(default_factory=lambda: [
        "lanes", "maxspeed_mph", "% Observed",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "x_km", "y_km",
    ])

    _ordinal_encoder: OrdinalEncoder | None = None
    _scaler: StandardScaler | None = None
    _num_median: Dict[str, float] = field(default_factory=dict)

    # ───────────────────────── geometry & time helpers ──────────────────────────
    @staticmethod
    def _add_xy(df: pd.DataFrame) -> pd.DataFrame:
        lon, lat = df["Longitude"].to_numpy(), df["Latitude"].to_numpy()
        if _TRANSFORMER:
            x_m, y_m = _TRANSFORMER.transform(lon, lat)
        else:
            x_m, y_m = _fallback_ll_to_xy(lon, lat)
        df["x_km"], df["y_km"] = x_m / 1_000.0, y_m / 1_000.0
        return df

    @staticmethod
    def _add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df["Time"], errors="coerce")
        hour = dt.dt.hour + dt.dt.minute / 60.0
        dow = dt.dt.dayofweek
        df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        df["dow_sin"], df["dow_cos"]   = np.sin(2 * np.pi * dow / 7),  np.cos(2 * np.pi * dow / 7)
        return df

    @staticmethod
    def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
        df["lanes"] = pd.to_numeric(df["lanes"], errors="coerce")
        df["maxspeed_mph"] = (
            df["maxspeed"].astype(str).str.extract(r"(\d+(?:\.\d+)?)").astype(float)
        )
        return df

    # ──────────────────────────── fit / transform ──────────────────────────────
    def fit(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> "TrafficDataEncoder":
        df = df.copy()
        df = self._add_xy(df)
        df = self._add_time_feats(df)
        df = self._clean_numeric(df)

        # fill NA before fitting encoders/scaler
        df[self._cat_cols] = df[self._cat_cols].fillna("UNK")
        self._num_median = df[self._num_cols].median().to_dict()
        df[self._num_cols] = df[self._num_cols].fillna(self._num_median)

        self._ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._ordinal_encoder.fit(df[self._cat_cols])

        self._scaler = StandardScaler()
        self._scaler.fit(df[self._num_cols])
        return self

    def _apply_preproc(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_xy(df)
        df = self._add_time_feats(df)
        df = self._clean_numeric(df)
        df[self._cat_cols] = df[self._cat_cols].fillna("UNK")
        df[self._num_cols] = df[self._num_cols].fillna(self._num_median)
        return df

    def transform(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> Tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["_ordinal_encoder", "_scaler", "_num_median"]) # type: ignore
        df = self._apply_preproc(df.copy())

        cat_arr = self._ordinal_encoder.transform(df[self._cat_cols]).astype(np.float32) # type: ignore
        num_arr = self._scaler.transform(df[self._num_cols]).astype(np.float32) # type: ignore
        feats   = np.concatenate([num_arr, cat_arr], axis=1)
        target  = df[target_col].to_numpy(dtype=np.float32)

        X, y = [], []
        for i in range(len(df) - self.seq_len - self.horizon + 1):
            X.append(feats[i : i + self.seq_len])
            y.append(target[i + self.seq_len : i + self.seq_len + self.horizon])
        return np.stack(X), np.stack(y)

    def fit_transform(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df, target_col).transform(df, target_col)


if __name__ == "__main__":
    import argparse, pathlib, joblib

    p = argparse.ArgumentParser(description="Encode 5‑min traffic CSV → numpy tensors")
    p.add_argument("csv_file", type=str)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--seq", type=int, default=12)
    p.add_argument("--h", type=int, default=1)
    args = p.parse_args()

    df_raw = pd.read_csv(args.csv_file)
    enc = TrafficDataEncoder(seq_len=args.seq, horizon=args.h).fit(df_raw)
    X, y = enc.transform(df_raw)
    print("Features shape:", X.shape, "Target shape:", y.shape)

    if args.save:
        joblib.dump(enc, pathlib.Path(args.save))
        print("Saved encoder →", args.save)


#in order to train right now I need to have the csv ordered by sensor_id 
#(same coordinates) and then for each sensor id to be ordered by the time in growing order