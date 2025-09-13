"""
predict_road.py – Predict traffic speeds for all sensors on a specific road and direction

This module provides functions to predict traffic speeds for all sensors on a given road
and direction at a specific time. Designed for map visualization and real-time prediction.
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import joblib
import sys
import os

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from encode import TrafficDataEncoder
from train_lstm import LSTMRegressor


class RoadPredictor:
    """
    Predictor for traffic speeds on specific roads and directions.
    
    This class loads a trained model and encoder, then provides methods to predict
    speeds for all sensors on a given road/direction at a specific time.
    """
    
    def __init__(self, model_path: str, encoder_path: str, device: str = "auto"):
        """
        Initialize the road predictor.
        
        Args:
            model_path: Path to trained model (.pt file)
            encoder_path: Path to fitted encoder (.pkl file)
            device: Device to use (auto, cpu, cuda, mps)
        """
        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load encoder
        print(f"Loading encoder from {encoder_path}")
        self.encoder = TrafficDataEncoder.load(encoder_path)
        
        # Load model
        print(f"Loading model from {model_path}")
        model_state = torch.load(model_path, map_location=self.device)
        
        # Infer model architecture from saved state
        n_features = len(self.encoder.num_cols) + len(self.encoder.cat_cols)
        
        # Infer hidden_size from first LSTM layer weights
        first_layer_weight_shape = model_state['lstm.weight_ih_l0'].shape
        hidden_size = first_layer_weight_shape[0] // 4
        
        # Check if bidirectional
        bidirectional = 'lstm.weight_ih_l0_reverse' in model_state
        
        # Infer number of layers
        layer_keys = [k for k in model_state.keys() if k.startswith('lstm.weight_ih_l')]
        n_layers = len(set([k.split('_l')[1].split('_')[0] for k in layer_keys]))
        
        print(f"Model architecture: hidden_size={hidden_size}, n_layers={n_layers}, bidirectional={bidirectional}")
        
        # Create and load model
        self.model = LSTMRegressor(
            n_features=n_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=0.3,
            bidirectional=bidirectional
        ).to(self.device)
        
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        print("Model and encoder loaded successfully")
    
    def get_road_sensors(self, df: pd.DataFrame, road_name: str, direction: str) -> pd.DataFrame:
        """
        Get all sensors for a specific road and direction.
        
        Args:
            df: DataFrame with traffic data
            road_name: Name of the road (e.g., "I 405")
            direction: Direction (e.g., "North", "South", "East", "West")
        
        Returns:
            DataFrame with unique sensors for the road/direction
        """
        # Filter for the specific road and direction
        road_data = df[(df['road_name'] == road_name) & (df['direction'] == direction)].copy()
        
        if len(road_data) == 0:
            raise ValueError(f"No data found for road '{road_name}' direction '{direction}'")
        
        # Get unique sensors using the actual sensor_id from the data
        sensors = road_data.groupby('sensor_id').agg({
            'Latitude': 'first',
            'Longitude': 'first',
            'road_name': 'first',
            'direction': 'first',
            'lanes': 'first'
        }).reset_index()
        
        print(f"Found {len(sensors)} sensors on {road_name} {direction}")
        return sensors
    
    def prepare_prediction_data(
        self, 
        df: pd.DataFrame, 
        road_name: str, 
        direction: str, 
        target_time: datetime,
        seq_len: int = 12
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for prediction by getting historical sequences for each sensor.
        
        Args:
            df: DataFrame with traffic data
            road_name: Name of the road
            direction: Direction
            target_time: Time to predict for
            seq_len: Length of historical sequence needed
        
        Returns:
            Tuple of (prepared_data, sensor_ids)
        """
        # Get sensors for this road/direction
        sensors = self.get_road_sensors(df, road_name, direction)
        
        # Prepare data for each sensor
        prepared_data = []
        sensor_ids = []
        
        for _, sensor in sensors.iterrows():
            sensor_id = sensor['sensor_id']
            
            # Get all data for this sensor
            sensor_data = df[df['sensor_id'] == sensor_id].copy()
            if len(sensor_data) == 0:
                print(f"Warning: No data found for sensor {sensor_id}")
                continue
            sensor_data = sensor_data.sort_values('Time').reset_index(drop=True)
            
            # Convert Time to datetime
            sensor_data['Time'] = pd.to_datetime(sensor_data['Time'])
            
            # Find the closest time to target_time
            time_diffs = abs(sensor_data['Time'] - target_time)
            if len(time_diffs) == 0:
                print(f"Warning: No time data for sensor {sensor_id}")
                continue
            closest_idx = time_diffs.idxmin()
            
            # Get historical sequence ending at closest time
            start_idx = max(0, closest_idx - seq_len + 1)
            end_idx = closest_idx + 1
            
            if end_idx - start_idx < seq_len:
                # Not enough historical data, skip this sensor
                print(f"Warning: Not enough historical data for sensor {sensor_id} (need {seq_len}, have {end_idx - start_idx})")
                continue
            
            # Get the sequence
            sequence_data = sensor_data.iloc[start_idx:end_idx].copy()
            
            # Ensure we have exactly seq_len points
            if len(sequence_data) > seq_len:
                sequence_data = sequence_data.tail(seq_len)
            
            # Verify we have the right number of points
            if len(sequence_data) != seq_len:
                print(f"Warning: Sequence length mismatch for sensor {sensor_id} (expected {seq_len}, got {len(sequence_data)})")
                continue
            
            # Add to prepared data
            prepared_data.append(sequence_data)
            sensor_ids.append(sensor_id)
        
        if not prepared_data:
            raise ValueError(f"No sensors with sufficient historical data for {road_name} {direction}")
        
        # Combine all sensor data and ensure proper sorting
        combined_data = pd.concat(prepared_data, ignore_index=True)
        
        # Ensure the data is sorted by sensor_id and Time (required by encoder)
        combined_data = combined_data.sort_values(['sensor_id', 'Time']).reset_index(drop=True)
        
        # Add time features that the encoder expects
        combined_data = self.encoder._add_time_features(combined_data)
        
        print(f"Prepared data for {len(sensor_ids)} sensors")
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Unique sensors in prepared data: {combined_data['sensor_id'].nunique()}")
        
        return combined_data, sensor_ids
    
    def predict_road_speeds(
        self, 
        df: pd.DataFrame, 
        road_name: str, 
        direction: str, 
        target_time: datetime
    ) -> pd.DataFrame:
        """
        Predict speeds for all sensors on a specific road and direction.
        
        Args:
            df: DataFrame with traffic data
            road_name: Name of the road (e.g., "I 405")
            direction: Direction (e.g., "North", "South", "East", "West")
            target_time: Time to predict for
        
        Returns:
            DataFrame with predictions for each sensor
        """
        print(f"Predicting speeds for {road_name} {direction} at {target_time}")
        
        # Prepare data
        prepared_data, sensor_ids = self.prepare_prediction_data(
            df, road_name, direction, target_time
        )
        
        # Instead of using the encoder's transform method, let's create sequences manually
        # since we already have the exact sequences we want
        print(f"Creating sequences manually from {len(prepared_data)} rows...")
        
        # Group by sensor and create sequences
        X_sequences = []
        y_targets = []
        sensor_mapping = []
        
        for sensor_id in sensor_ids:
            sensor_data = prepared_data[prepared_data['sensor_id'] == sensor_id].sort_values('Time')
            
            if len(sensor_data) >= self.encoder.seq_len:
                # Get the last seq_len points as input sequence
                sequence_data = sensor_data.tail(self.encoder.seq_len)
                
                # Prepare features
                cat_features = self.encoder.ordinal_encoder.transform(sequence_data[self.encoder.cat_cols]).astype(np.float32)
                num_features = self.encoder.scaler.transform(sequence_data[self.encoder.num_cols]).astype(np.float32)
                features = np.concatenate([num_features, cat_features], axis=1)
                
                X_sequences.append(features)
                
                # For prediction, we don't have a target, so we'll use the last speed as placeholder
                last_speed = sequence_data[self.encoder.target_col].iloc[-1]
                y_targets.append([last_speed])
                
                sensor_mapping.append(sensor_id)
        
        if not X_sequences:
            raise ValueError("No valid sequences found for prediction")
        
        X = np.stack(X_sequences, axis=0)
        y = np.array(y_targets)
        
        print(f"Created {len(X_sequences)} sequences with shape {X.shape}")
        
        # Make predictions for each sequence
        predictions = []
        sensor_info = []
        
        for i, sensor_id in enumerate(sensor_mapping):
            # Get the sequence for this sensor
            sensor_sequence = X[i:i+1]  # Keep batch dimension
            
            # Make prediction
            with torch.no_grad():
                sensor_sequence_tensor = torch.from_numpy(sensor_sequence).float().to(self.device)
                prediction = self.model(sensor_sequence_tensor).cpu().numpy()[0, 0]
            
            predictions.append(prediction)
            
            # Get sensor info
            sensor_data = prepared_data[prepared_data['sensor_id'] == sensor_id].iloc[0]
            
            # Get real speed from the most recent data point
            real_speed = sensor_data.get('speed_mph', None)
            if real_speed is None and 'speed' in sensor_data:
                real_speed = sensor_data['speed']
            elif real_speed is None and 'Speed' in sensor_data:
                real_speed = sensor_data['Speed']
            
            sensor_info.append({
                'sensor_id': sensor_id,
                'Latitude': sensor_data['Latitude'],
                'Longitude': sensor_data['Longitude'],
                'road_name': sensor_data['road_name'],
                'direction': sensor_data['direction'],
                'lanes': sensor_data['lanes'],
                'predicted_speed': prediction,
                'real_speed': real_speed,
                'target_time': target_time
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(sensor_info)
        
        print(f"Generated predictions for {len(results_df)} sensors")
        print(f"Predicted speed range: {results_df['predicted_speed'].min():.1f} - {results_df['predicted_speed'].max():.1f} mph")
        
        # Print real speed statistics if available
        real_speeds = results_df['real_speed'].dropna()
        if len(real_speeds) > 0:
            print(f"Real speed range: {real_speeds.min():.1f} - {real_speeds.max():.1f} mph")
            print(f"Real speed available for {len(real_speeds)}/{len(results_df)} sensors")
        else:
            print("No real speed data available")
        
        return results_df
    
    def predict_multiple_times(
        self, 
        df: pd.DataFrame, 
        road_name: str, 
        direction: str, 
        target_times: List[datetime]
    ) -> pd.DataFrame:
        """
        Predict speeds for multiple time points.
        
        Args:
            df: DataFrame with traffic data
            road_name: Name of the road
            direction: Direction
            target_times: List of times to predict for
        
        Returns:
            DataFrame with predictions for all sensors at all times
        """
        all_predictions = []
        
        for target_time in target_times:
            try:
                predictions = self.predict_road_speeds(df, road_name, direction, target_time)
                all_predictions.append(predictions)
            except Exception as e:
                print(f"Error predicting for {target_time}: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("No successful predictions generated")
        
        # Combine all predictions
        combined_df = pd.concat(all_predictions, ignore_index=True)
        
        return combined_df


def predict_road_speeds(
    df: pd.DataFrame,
    road_name: str,
    direction: str,
    target_time: datetime,
    model_path: str,
    encoder_path: str,
    device: str = "auto"
) -> pd.DataFrame:
    """
    Convenience function to predict speeds for all sensors on a road.
    
    Args:
        df: DataFrame with traffic data
        road_name: Name of the road (e.g., "I 405")
        direction: Direction (e.g., "North", "South", "East", "West")
        target_time: Time to predict for
        model_path: Path to trained model (.pt file)
        encoder_path: Path to fitted encoder (.pkl file)
        device: Device to use (auto, cpu, cuda, mps)
    
    Returns:
        DataFrame with predictions for each sensor
    """
    predictor = RoadPredictor(model_path, encoder_path, device)
    return predictor.predict_road_speeds(df, road_name, direction, target_time)


def main():
    """Example usage of the road predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict traffic speeds for a specific road and direction")
    parser.add_argument("--csv", required=True, help="Path to CSV file with traffic data")
    parser.add_argument("--model", required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--encoder", required=True, help="Path to fitted encoder (.pkl file)")
    parser.add_argument("--road", required=True, help="Road name (e.g., 'I 405')")
    parser.add_argument("--direction", required=True, help="Direction (e.g., 'North', 'South', 'East', 'West')")
    parser.add_argument("--time", required=True, help="Target time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--output", help="Path to save predictions CSV")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv}")
    df = pd.read_csv(args.csv)
    
    # Parse target time
    target_time = datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
    
    # Make predictions
    predictions = predict_road_speeds(
        df, args.road, args.direction, target_time, 
        args.model, args.encoder
    )
    
    # Print results
    print(f"\nPredictions for {args.road} {args.direction} at {target_time}:")
    print("=" * 60)
    for _, row in predictions.iterrows():
        print(f"Sensor {row['sensor_id']}: {row['predicted_speed']:.1f} mph")
    
    # Save if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()
