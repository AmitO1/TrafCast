import pandas as pd
import numpy as np

class MockTrafficPredictor:
    def __init__(self, road_classification_map: dict, seed: int = 42):
        """
        road_classification_map: dict mapping road ref (e.g., 'I 405') to classification ('busy', 'moderate', 'free')
        """
        valid_classes = {'busy', 'moderate', 'free'}
        for cls in road_classification_map.values():
            if cls not in valid_classes:
                raise ValueError(f"Invalid classification '{cls}', must be one of {valid_classes}")
        self.road_classification_map = road_classification_map
        self.random = np.random.default_rng(seed)

        self.speed_range = {
            'busy': (0.2, 0.5),
            'moderate': (0.5, 0.8),
            'free': (0.8, 1.0)
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'ref' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'ref' column with road names")

        road_ref = df['ref'].iloc[0]
        classification = self.road_classification_map.get(road_ref, 'moderate')
        min_r, max_r = self.speed_range[classification]

        df['maxspeed_numeric'] = df['maxspeed'].str.extract(r'(\d+)').astype(float)

        # Generate base values with slight spatial variation
        base = self.random.uniform(min_r, max_r)
        noise = self.random.normal(loc=0, scale=0.05, size=len(df))  # small gaussian noise
        raw_factors = np.clip(base + noise, min_r, max_r)

        df['speed'] = df['maxspeed_numeric'] * raw_factors

        return df.drop(columns=['maxspeed_numeric'])
