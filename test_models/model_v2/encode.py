from __future__ import annotations
"""
encode.py – robust, sensor-safe feature encoder for 5-minute traffic data with
stride, per-sensor caps, and class-aware downsampling. Returns the exact target
row indices and timestamps used for each window to keep downstream mapping correct.

Pipeline
--------
1.  Sensor key + sorting   build a stable sensor_id and sort by (sensor_id, Time)
2.  Geographic projection  (lat, lon) → local x_km, y_km (EPSG:6423, with fallback)
3.  Time features          sin/cos hour-of-day & day-of-week
4.  Numeric clean-up       lanes → float, maxspeed → float mph, % Observed → float
5.  Missing-value fill     categorical → "UNK", numeric → median
6.  Encoding               OrdinalEncoder + StandardScaler
7.  Sliding windows        **within each sensor only** (no cross-sensor leakage)

Notes
-----
- Windows **never** cross sensor boundaries.
- Designed to be fitted once (on training) and reused for inference.
- Returns (X, y, tgt_rows, tgt_times) so callers can align meta/timestamps.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

# ─────────────────────────── geo transform setup ─────────────────────────────
try:
    import pyproj
    _TRANSFORMER = pyproj.Transformer.from_crs("epsg:4326", "epsg:6423", always_xy=True)
except Exception:
    _TRANSFORMER = None

def _fallback_ll_to_xy(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple equirectangular fallback (approximate, but stable)."""
    R = 6_371_000.0  # metres
    x = np.deg2rad(lon) * R * np.cos(np.deg2rad(float(np.nanmean(lat))))
    y = np.deg2rad(lat) * R
    return x, y

__all__ = ["TrafficDataEncoder"]


@dataclass
class TrafficDataEncoder:
    horizon: int = 1
    seq_len: int = 12

    # Downsampling knobs
    stride: int = 1
    max_windows_per_sensor: int | None = None
    keep_prob_low: float = 1.0
    keep_prob_med: float = 1.0
    keep_prob_high: float = 1.0
    rng_seed: int = 42

    # Columns
    _cat_cols: List[str] = field(default_factory=lambda: ["direction", "weather"])
    _num_cols: List[str] = field(default_factory=lambda: [
        "lanes", "maxspeed_mph", "% Observed",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "x_km", "y_km",
    ])

    # Fitted artefacts
    _ordinal_encoder: OrdinalEncoder | None = None
    _scaler: StandardScaler | None = None
    _num_median: Dict[str, float] = field(default_factory=dict)

    # ───────────────────────── helpers: id/sort, geo, time, numeric ──────────
    @staticmethod
    def _ensure_sensor_id_and_sort(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "sensor_id" not in df.columns:
            df["sensor_id"] = (
                df["Latitude"].round(6).astype(str) + ";" + df["Longitude"].round(6).astype(str)
            )
        # Parse to datetime once, in place
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        return df.sort_values(["sensor_id", "Time"]).reset_index(drop=True)

    @staticmethod
    def _add_xy(df: pd.DataFrame) -> pd.DataFrame:
        lon, lat = df["Longitude"].to_numpy(), df["Latitude"].to_numpy()
        if _TRANSFORMER is not None:
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
        df["dow_sin"], df["dow_cos"]   = np.sin(2 * np.pi * dow / 7),   np.cos(2 * np.pi * dow / 7)
        return df

    @staticmethod
    def _clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
        # lanes may come as strings
        df["lanes"] = pd.to_numeric(df.get("lanes"), errors="coerce")
        # % Observed may be string
        df["% Observed"] = pd.to_numeric(df.get("% Observed"), errors="coerce")
        # Extract first number from maxspeed like "65 mph" → 65.0
        maxs = df.get("maxspeed")
        if maxs is not None:
            df["maxspeed_mph"] = (
                maxs.astype(str).str.extract(r"(\d+(?:\.\d+)?)").astype(float)
            )
        else:
            df["maxspeed_mph"] = np.nan
        return df

    # ──────────────────────────── fit / transform ─────────────────────────────
    def fit(self, df: pd.DataFrame, target_col: str = "AggSpeed") -> "TrafficDataEncoder":
        df = self._ensure_sensor_id_and_sort(df)
        df = self._add_xy(df)
        df = self._add_time_feats(df)
        df = self._clean_numeric(df)

        # Missing-value handling BEFORE fitting encoders/scalers
        df[self._cat_cols] = df[self._cat_cols].fillna("UNK")
        self._num_median = df[self._num_cols].median(numeric_only=True).to_dict()
        df[self._num_cols] = df[self._num_cols].fillna(self._num_median)

        self._ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._ordinal_encoder.fit(df[self._cat_cols])

        self._scaler = StandardScaler()
        self._scaler.fit(df[self._num_cols])
        return self

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_sensor_id_and_sort(df)
        df = self._add_xy(df)
        df = self._add_time_feats(df)
        df = self._clean_numeric(df)
        df[self._cat_cols] = df[self._cat_cols].fillna("UNK")
        df[self._num_cols] = df[self._num_cols].fillna(self._num_median)
        return df

    def transform(
        self, df: pd.DataFrame, target_col: str = "AggSpeed"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X : (N, seq_len, F)
        y : (N, horizon)
        tgt_rows : (N,)  integer row indices in the *preprocessed* df
        tgt_times: (N,)  datetime64[ns] timestamps corresponding to tgt_rows
        """
        check_is_fitted(self, ["_ordinal_encoder", "_scaler", "_num_median"])  # type: ignore
        df = self._preprocess(df)

        feat_dim = len(self._num_cols) + len(self._cat_cols)
        X_chunks: List[np.ndarray] = []
        y_chunks: List[np.ndarray] = []
        tgt_rows: List[int] = []

        rng = np.random.default_rng(self.rng_seed)

        for _, g in df.groupby("sensor_id", sort=False):
            n = len(g)
            if n < self.seq_len + self.horizon:
                continue

            cat_arr = self._ordinal_encoder.transform(g[self._cat_cols]).astype(np.float32)  # type: ignore
            num_arr = self._scaler.transform(g[self._num_cols]).astype(np.float32)           # type: ignore
            feats   = np.concatenate([num_arr, cat_arr], axis=1)
            target  = g[target_col].to_numpy(dtype=np.float32)

            limit = n - self.seq_len - self.horizon + 1
            idx = np.arange(0, limit, max(1, self.stride), dtype=int)

            # Optional per-sensor cap after stride (preserve chronology by even-spacing)
            if self.max_windows_per_sensor is not None and len(idx) > self.max_windows_per_sensor:
                idx = np.linspace(0, len(idx)-1, num=self.max_windows_per_sensor, dtype=int)

            # Class-aware keep probabilities (decide using the horizon target)
            if (self.keep_prob_low < 1.0) or (self.keep_prob_med < 1.0) or (self.keep_prob_high < 1.0):
                low_t, high_t = 35.0, 55.0
                y_h = target[idx + self.seq_len + self.horizon - 1]
                p = np.where(
                    y_h <= low_t, self.keep_prob_low,
                    np.where(y_h <= high_t, self.keep_prob_med, self.keep_prob_high)
                )
                keep_mask = rng.random(len(idx)) < p
                idx = idx[keep_mask]

            for i in idx:
                X_chunks.append(feats[i : i + self.seq_len])
                y_chunks.append(target[i + self.seq_len : i + self.seq_len + self.horizon])
                tgt_rows.append(int(g.index[i + self.seq_len + self.horizon - 1]))

        if not X_chunks:
            return (
                np.empty((0, self.seq_len, feat_dim), dtype=np.float32),
                np.empty((0, self.horizon), dtype=np.float32),
                np.empty((0,), dtype=int),
                np.empty((0,), dtype="datetime64[ns]"),
            )

        X = np.stack(X_chunks, axis=0)
        y = np.stack(y_chunks, axis=0)
        tgt_rows_arr = np.asarray(tgt_rows, dtype=int)
        tgt_times = df.loc[tgt_rows_arr, "Time"].to_numpy()

        return X, y, tgt_rows_arr, tgt_times

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "AggSpeed"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.fit(df, target_col).transform(df, target_col)


# ─────────────────────────────── CLI utility (optional) ────────────────────────
if __name__ == "__main__":
    import argparse, pathlib, joblib

    p = argparse.ArgumentParser(description="Encode 5-min traffic CSV → numpy tensors (sensor-safe)")
    p.add_argument("csv_file", type=str, help="Path to raw CSV")
    p.add_argument("--save", type=str, default=None, help="Optional path to save fitted encoder (joblib .pkl)")
    p.add_argument("--seq", type=int, default=12, help="Sequence length (history steps)")
    p.add_argument("--h", type=int, default=1, help="Horizon (steps ahead)")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_windows_per_sensor", type=int, default=None)
    p.add_argument("--keep_prob_low", type=float, default=1.0)
    p.add_argument("--keep_prob_med", type=float, default=1.0)
    p.add_argument("--keep_prob_high", type=float, default=1.0)
    args = p.parse_args()

    df_raw = pd.read_csv(args.csv_file)
    enc = TrafficDataEncoder(
        seq_len=args.seq, horizon=args.h,
        stride=args.stride, max_windows_per_sensor=args.max_windows_per_sensor,
        keep_prob_low=args.keep_prob_low, keep_prob_med=args.keep_prob_med, keep_prob_high=args.keep_prob_high
    ).fit(df_raw)
    X, y, tgt_rows, tgt_times = enc.transform(df_raw)
    print("Features:", X.shape, "| Target:", y.shape, "| Windows:", len(tgt_rows))
    if args.save:
        out = pathlib.Path(args.save)
        if out.suffix != ".pkl":
            out = out.with_suffix(".pkl")
        joblib.dump(enc, out)
        print("Saved encoder →", str(out))
