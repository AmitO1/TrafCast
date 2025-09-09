"""
encode.py – Traffic data encoder for LSTM traffic flow prediction

This module provides TrafficDataEncoder for processing 5-minute traffic sensor data
into sequences suitable for LSTM training. Key features:
- Sensor-safe windowing (no cross-sensor leakage)
- Feature engineering (time, geographic, categorical)
- Speed-based class weighting support
- Robust missing value handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from typing import List, Tuple, Dict, Optional
import joblib
from pathlib import Path


class TrafficDataEncoder:
    """
    Encodes traffic sensor data into sequences for LSTM training.
    
    Features:
    - Geographic coordinates (lat/lon -> x/y km)
    - Time features (hour, day of week)
    - Categorical encoding (direction, weather)
    - Speed-based class weighting
    - Sensor-safe windowing
    """
    
    def __init__(
        self,
        seq_len: int = 12,  # 12 * 5min = 1 hour history
        horizon: int = 1,   # predict 1 step ahead (5 minutes)
        target_col: str = "speed_mph"
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_col = target_col
        
        # Feature columns
        self.cat_cols = ["direction", "weather"]
        self.num_cols = [
            "lanes", "% Observed", "Latitude", "Longitude",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ]
        
        # Fitted components
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.num_medians: Dict[str, float] = {}
        self.is_fitted = False
    
    def _ensure_sensor_id_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sensor_id and sort by sensor and time."""
        df = df.copy()
        
        # Create sensor_id from coordinates
        if "sensor_id" not in df.columns:
            df["sensor_id"] = (
                df["Latitude"].round(6).astype(str) + ";" + 
                df["Longitude"].round(6).astype(str)
            )
        
        # Parse time and sort
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        return df.sort_values(["sensor_id", "Time"]).reset_index(drop=True)
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time features."""
        dt = pd.to_datetime(df["Time"], errors="coerce")
        hour = dt.dt.hour + dt.dt.minute / 60.0
        dow = dt.dt.dayofweek
        
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        
        return df
    
    def _clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert numeric columns."""
        # Ensure lanes is numeric
        df["lanes"] = pd.to_numeric(df.get("lanes", 0), errors="coerce")
        
        # Ensure % Observed is numeric
        df["% Observed"] = pd.to_numeric(df.get("% Observed", 100), errors="coerce")
        
        return df
    
    def _compute_speed_weights(self, y: np.ndarray) -> Dict[str, float]:
        """Compute class weights for speed-based weighting."""
        # Define speed classes based on user's experience
        low_mask = y <= 30
        high_mask = y >= 60
        medium_mask = ~(low_mask | high_mask)
        
        n_low = low_mask.sum()
        n_medium = medium_mask.sum()
        n_high = high_mask.sum()
        n_total = len(y)
        
        print(f"Speed distribution:")
        print(f"  Low (≤30): {n_low} samples ({n_low/n_total*100:.1f}%)")
        print(f"  Medium (30-60): {n_medium} samples ({n_medium/n_total*100:.1f}%)")
        print(f"  High (≥60): {n_high} samples ({n_high/n_total*100:.1f}%)")
        
        # Compute inverse frequency weights
        if n_low > 0 and n_medium > 0 and n_high > 0:
            weight_low = n_total / (3 * n_low)
            weight_medium = n_total / (3 * n_medium)
            weight_high = n_total / (3 * n_high)
        else:
            weight_low = weight_medium = weight_high = 1.0
        
        print(f"Class weights: Low={weight_low:.2f}, Medium={weight_medium:.2f}, High={weight_high:.2f}")
        
        return {
            "weight_low": weight_low,
            "weight_medium": weight_medium,
            "weight_high": weight_high,
            "low_threshold": 30,
            "high_threshold": 60
        }
    
    def fit(self, df: pd.DataFrame) -> "TrafficDataEncoder":
        """Fit the encoder on training data."""
        print("Fitting encoder...")
        
        # Preprocess data
        df = self._ensure_sensor_id_and_sort(df)
        df = self._add_time_features(df)
        df = self._clean_numeric(df)
        
        # Handle missing values
        df[self.cat_cols] = df[self.cat_cols].fillna("UNK")
        self.num_medians = df[self.num_cols].median(numeric_only=True).to_dict()
        df[self.num_cols] = df[self.num_cols].fillna(self.num_medians)
        
        # Fit encoders
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", 
            unknown_value=-1
        )
        self.ordinal_encoder.fit(df[self.cat_cols])
        
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.num_cols])
        
        self.is_fitted = True
        print("Encoder fitted successfully")
        return self
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps."""
        df = self._ensure_sensor_id_and_sort(df)
        df = self._add_time_features(df)
        df = self._clean_numeric(df)
        
        # Handle missing values using fitted medians
        df[self.cat_cols] = df[self.cat_cols].fillna("UNK")
        df[self.num_cols] = df[self.num_cols].fillna(self.num_medians)
        
        return df
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform data into sequences.
        
        Returns:
            X: (N, seq_len, n_features) - input sequences
            y: (N, horizon) - target values
            target_indices: (N,) - indices of target rows in original df
            timestamps: (N,) - timestamps of target rows
        """
        check_is_fitted(self, ["ordinal_encoder", "scaler", "num_medians"])
        
        df = self._preprocess(df)
        
        X_chunks = []
        y_chunks = []
        target_indices = []
        timestamps = []
        
        # Process each sensor separately to avoid cross-sensor leakage
        for sensor_id, group in df.groupby("sensor_id", sort=False):
            if len(group) < self.seq_len + self.horizon:
                continue  # Not enough data for this sensor
            
            # Encode features
            cat_features = self.ordinal_encoder.transform(group[self.cat_cols]).astype(np.float32)
            num_features = self.scaler.transform(group[self.num_cols]).astype(np.float32)
            features = np.concatenate([num_features, cat_features], axis=1)
            
            # Get targets
            targets = group[self.target_col].to_numpy(dtype=np.float32)
            group_timestamps = group["Time"].to_numpy()
            group_indices = group.index.to_numpy()
            
            # Create sliding windows
            n_windows = len(group) - self.seq_len - self.horizon + 1
            for i in range(n_windows):
                X_chunks.append(features[i:i + self.seq_len])
                y_chunks.append(targets[i + self.seq_len:i + self.seq_len + self.horizon])
                target_indices.append(group_indices[i + self.seq_len + self.horizon - 1])
                timestamps.append(group_timestamps[i + self.seq_len + self.horizon - 1])
        
        if not X_chunks:
            # Return empty arrays with correct shapes
            n_features = len(self.num_cols) + len(self.cat_cols)
            return (
                np.empty((0, self.seq_len, n_features), dtype=np.float32),
                np.empty((0, self.horizon), dtype=np.float32),
                np.empty(0, dtype=int),
                np.empty(0, dtype=object)
            )
        
        X = np.stack(X_chunks, axis=0)
        y = np.stack(y_chunks, axis=0)
        target_indices = np.array(target_indices, dtype=int)
        timestamps = np.array(timestamps)
        
        print(f"Created {len(X)} sequences from {len(df.groupby('sensor_id'))} sensors")
        return X, y, target_indices, timestamps
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit encoder and transform data in one step."""
        return self.fit(df).transform(df)
    
    def get_speed_weights(self, y: np.ndarray) -> Dict[str, float]:
        """Get speed-based class weights for weighted loss."""
        return self._compute_speed_weights(y)
    
    def save(self, filepath: str) -> None:
        """Save the fitted encoder."""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before saving")
        
        joblib.dump(self, filepath)
        print(f"Encoder saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "TrafficDataEncoder":
        """Load a fitted encoder."""

        try:
            encoder = joblib.load(filepath)
            if not isinstance(encoder, cls):
                raise ValueError(f"Loaded object is not a {cls.__name__}")
            return encoder
        except AttributeError as e:
            if "TrafficDataEncoder" in str(e):
                # Handle the case where encoder was saved from a different module context
                print("Warning: Encoder was saved from different module context. Reconstructing...")
                
                # Use a more robust approach with joblib
                import sys
                import types
                
                # Temporarily modify sys.modules to include our class
                original_main = sys.modules.get('__main__')
                temp_module = types.ModuleType('temp_encode')
                temp_module.TrafficDataEncoder = cls
                sys.modules['__main__'] = temp_module
                
                try:
                    # Now try loading with the modified module context
                    encoder = joblib.load(filepath)
                    if not isinstance(encoder, cls):
                        raise ValueError(f"Loaded object is not a {cls.__name__}")
                    return encoder
                finally:
                    # Restore original __main__ module
                    if original_main is not None:
                        sys.modules['__main__'] = original_main
                    else:
                        del sys.modules['__main__']
            else:
                raise e


def main():
    """CLI interface for encoding data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Encode traffic data for LSTM training")
    parser.add_argument("csv_file", help="Path to CSV file with traffic data")
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence length (default: 12)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (default: 1)")
    parser.add_argument("--target_col", default="speed_mph", help="Target column name")
    parser.add_argument("--save_encoder", help="Path to save fitted encoder")
    parser.add_argument("--output", help="Path to save encoded data (optional)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    # Create and fit encoder
    encoder = TrafficDataEncoder(
        seq_len=args.seq_len,
        horizon=args.horizon,
        target_col=args.target_col
    )
    
    X, y, target_indices, timestamps = encoder.fit_transform(df)
    
    print(f"Encoded data shapes:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Target indices: {len(target_indices)}")
    print(f"  Timestamps: {len(timestamps)}")
    
    # Save encoder if requested
    if args.save_encoder:
        encoder.save(args.save_encoder)
    
    # Save encoded data if requested
    if args.output:
        np.savez(args.output, X=X, y=y, target_indices=target_indices, timestamps=timestamps)
        print(f"Encoded data saved to {args.output}")


if __name__ == "__main__":
    main()

