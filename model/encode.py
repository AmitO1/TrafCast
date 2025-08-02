from __future__ import annotations
"""encode.py – feature engineering for 5‑minute traffic speed data.

Given a pandas DataFrame with columns that match the raw LA sensor dump

    Latitude  Longitude  lanes  maxspeed  ref  direction  road_name  Time  AggSpeed  % Observed  weather

this module converts it into a model‑ready tensor *without* using
`sensor_id`.  Steps:

1. **Geographic projection**  (lat, lon) → local x_km, y_km (EPSG:6423)
2. **Time features**          sin/cos hour‑of‑day, day‑of‑week, weekend
3. **Numeric clean‑up**       lanes → int, maxspeed → mph as float
4. **Categorical encoding**   direction, road_name, weather → ordinal ints
5. **Scaling**                StandardScaler on numerics (fit on train only)
6. **Windowing**              optional sliding window builder for LSTM

"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    # NAD83 / California zone (metres)
    import pyproj

    _TRANSFORMER = pyproj.Transformer.from_crs("epsg:4326", "epsg:6423", always_xy=True)
except Exception:  # pyproj missing or other error ⇒ fallback

    def _fallback_ll_to_xy(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple equirectangular projection ~metres (not for production)."""
        R = 6_371_000  # Earth radius (m)
        x = np.deg2rad(lon) * R * np.cos(np.deg2rad(lat.mean()))
        y = np.deg2rad(lat) * R
        return x, y

    _TRANSFORMER = None

__all__ = [
    "TrafficDataEncoder",
]


@dataclass
class TrafficDataEncoder:
    """Fit‑transform utility that returns (X, y) NumPy arrays ready for torch/TF."""

    horizon: int = 1  # predict 1 step (5‑min) ahead
    seq_len: int = 12  # history window (12×5=60 min)
    _cat_cols: List[str] = field(default_factory=lambda: ["direction", "road_name", "weather"])
    _num_cols: List[str] = field(default_factory=lambda: [
        "lanes",
        "maxspeed_mph",
        "% Observed",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "x_km",
        "y_km",
    ])
    _ordinal_encoder: OrdinalEncoder | None = None
    _scaler: StandardScaler | None = None

    # ───────────────────────────── preprocessing helpers ──────────────────────────

    @staticmethod
    def _add_xy(df: pd.DataFrame) -> pd.DataFrame:
        lon = df["Longitude"].to_numpy()
        lat = df["Latitude"].to_numpy()
        if _TRANSFORMER is not None:
            x_m, y_m = _TRANSFORMER.transform(lon, lat)
        else:  # crude fallback
            x_m, y_m = _fallback_ll_to_xy(lon, lat)
        df["x_km"] = x_m / 1000.0
        df["y_km"] = y_m / 1000.0
        return df

    @staticmethod
    def _add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
        dt = pd.to_datetime(df["Time"], utc=False)
        hour = dt.dt.hour + dt.dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        dow = dt.dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        return df

    @staticmethod
    def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
        # Normalize 'lanes' values to be integers or NaN
        def normalize_lanes(value):
            if isinstance(value, list):
                try:
                    return min(int(x) for x in value)
                except Exception:
                    return np.nan
            try:
                return int(value)
            except Exception:
                return np.nan

        df["lanes"] = df["lanes"].apply(normalize_lanes)
        df["lanes"] = pd.to_numeric(df["lanes"], errors="coerce").astype("Int64")  # nullable int

        # Strip " mph" in maxspeed and extract numeric part
        df["maxspeed_mph"] = (
            df["maxspeed"]
            .astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
            .astype(float)
        )

        return df

    # ───────────────────────────── public API ────────────────────────────────────

    def fit(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> "TrafficDataEncoder":
        """Learn ordinal categories + numeric scaler on *df* (train split)."""
        df = df.copy()
        df = self._add_xy(df)
        df = self._add_time_feats(df)
        df = self._clean_numeric(df)

        self._ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self._ordinal_encoder.fit(df[self._cat_cols])

        self._scaler = StandardScaler()
        self._scaler.fit(df[self._num_cols])
        return self

    def transform(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> Tuple[np.ndarray, np.ndarray]:
        """Apply encoders, build windows, return (X, y) tensors."""
        check_is_fitted(self, ["_ordinal_encoder", "_scaler"]) # type: ignore
        df = df.copy()
        df = self._add_xy(df)
        df = self._add_time_feats(df)
        df = self._clean_numeric(df)

        # Encode categoricals and numerics
        cat_arr = self._ordinal_encoder.transform(df[self._cat_cols]).astype(np.float32) # type: ignore
        num_arr = self._scaler.transform(df[self._num_cols]).astype(np.float32) # type: ignore

        # combine feature columns
        feats = np.concatenate([num_arr, cat_arr], axis=1)  # (N, F)
        target = df[target_col].to_numpy(dtype=np.float32)

        # sliding windows → shape (M, seq_len, F)
        X, y = [], []
        for i in range(len(df) - self.seq_len - self.horizon + 1):
            X.append(feats[i : i + self.seq_len])
            y.append(target[i + self.seq_len : i + self.seq_len + self.horizon])
        return np.stack(X), np.stack(y)

    # convenience one‑liner
    def fit_transform(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df, target_col).transform(df, target_col)


if __name__ == "__main__":
    # demo on a small CSV path passed via CLI argument
    import argparse, pathlib, joblib

    p = argparse.ArgumentParser()
    p.add_argument("csv_file", type=str, help="path to raw traffic csv")
    p.add_argument("--save", type=str, default=None, help="pickle encoder to this path")
    args = p.parse_args()

    df_raw = pd.read_csv(args.csv_file)
    enc = TrafficDataEncoder()
    X, y = enc.fit_transform(df_raw)
    print("Features shape:", X.shape, "Target shape:", y.shape)
    if args.save is not None:
        joblib.dump(enc, pathlib.Path(args.save))
        print("Saved encoder →", args.save)
