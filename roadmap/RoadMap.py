import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
from shapely.geometry import LineString, MultiLineString, Point
from shapely import ops
from sklearn.neighbors import BallTree
from typing import Tuple, List
from .utils import get_coordinates_from_network, sort_gps_by_greedy_path, add_weather_to_df
from .mock_predictor import MockTrafficPredictor
from geopy.distance import geodesic
import re
import math
from datetime import datetime
import sys
import os
# Add parent directory to path to import model_v3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v3.predict_road import RoadPredictor

# Define model paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model_v3", "final_lstm.pt")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "model_v3", "final_encoder.pkl")

# Validate that model files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder file not found at: {ENCODER_PATH}")

DIST_THRESHOLD_METERS_MAX = 1200 #2000
DIST_THRESHOLD_METERS_MIN = 10 #10


class RoadMapManager:

    def __init__(self, city: str,bbox: Tuple[float,float,float,float], base_data_dir: str = "data"):
        self.city = city
        self.bbox = bbox
        self.base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", base_data_dir))
        self.city_path = os.path.join(self.base_data_dir, self.city)
        self.coordinates_path = os.path.join(self.city_path, 'coordinates') 
        self.roads_path = os.path.join(self.city_path, 'roads')
        self.road_network_path = os.path.join(self.city_path, 'maps')
        self.visualizations_path = os.path.join(self.city_path, 'visualizations')

        self.roads = dict()

        self._validate_structure()
        self._load_road_network(self.bbox)

        
    def _validate_structure(self):
        for path in [self.coordinates_path, self.roads_path, self.road_network_path, self.visualizations_path]:
            os.makedirs(path, exist_ok=True)
            
    @staticmethod
    def split_road_name_direction(road_name: str) -> Tuple[str, str]:
        parts = road_name.split()
        return " ".join(parts[:-1]), parts[-1]
    
    def set_roads(self, roads: List[str]):
        os.makedirs(self.coordinates_path, exist_ok=True)

        for road in roads:
            road_name, direction = self.split_road_name_direction(road)
            file_name = f"{road_name} {direction}.csv"
            file_path =  os.path.join(self.coordinates_path,file_name)
            
            if os.path.exists(file_path):
                print(f"DataFrame for {road_name} - {direction} already exists")
                df = pd.read_csv(file_path)
            else:
                print(f"Downloading DataFrame for {road_name} - {direction}")
                df = get_coordinates_from_network(self.road_network, road_name, direction)
                df.to_csv(file_path, index=False)

            self.roads[(road_name,direction)] = df  

    def get_roads(self):
        """
        Used for testing 
        """ 
        for (road_name, direction), df in self.roads.items():
            print(f"road name: {road_name} - {direction}")
            print(df.head())
            print("\n" + "-"*40 + "\n")  

    def _load_road_network(self, bbox: Tuple[float,float,float,float]):
        network_filename = f"{self.city.replace(' ', "_")}_network.graphml"
        network_path = os.path.join(self.road_network_path, network_filename)

        if os.path.exists(network_path):
            print("map already exists")
            self.road_network = ox.load_graphml(network_path)
        else:
            print("Downloading map")
            self.road_network = ox.graph_from_bbox(
                bbox=bbox,
                network_type='drive'
            )

            self.road_network = ox.bearing.add_edge_bearings(self.road_network)

            ox.save_graphml(self.road_network, filepath=network_path)

    def apply_prediction_data(self, predict_time: datetime | None = None):
        """
        Needed data to predict: 
        Gather data about weather in current day in time for each point
        Speed limit - from the road network
        Road name, Direction - key of the coordinates dict
        Cooridnate from the coordinate df
        Lanes - either from coordinate if given else from road network
        Time - input from user
        """
        predictions = {}
        road_predictor = RoadPredictor(MODEL_PATH, ENCODER_PATH)
        for (road_name, direction) in self.roads.keys():
            road_under = road_name.replace(" ", "_")
            df = pd.read_csv(os.path.join(self.roads_path, f"{road_under}_{direction.lower()}.csv.gz"), compression='gzip')
            predictions[(road_name, direction)] = road_predictor.predict_road_speeds(df, road_name, direction, predict_time)
        
        # Map predictions to road coordinates
        self._map_predictions_to_roads(predictions)

        for (road_name, direction), df in self.roads.items():
            print(df.head())
            df = sort_gps_by_greedy_path(df)
            self.roads[(road_name, direction)] = df    

    def _map_predictions_to_roads(self, predictions: dict):
        """
        Map predicted speeds to the closest points in self.roads coordinates.
        
        Args:
            predictions: Dictionary with (road_name, direction) keys and prediction DataFrames as values
        """
        from sklearn.neighbors import BallTree
        
        for (road_name, direction), road_df in self.roads.items():
            if (road_name, direction) not in predictions:
                print(f"No predictions found for {road_name} {direction}")
                continue
                
            pred_df = predictions[(road_name, direction)]
            
            if pred_df.empty:
                print(f"Empty predictions for {road_name} {direction}")
                continue
            
            # Extract coordinates from road data
            road_coords = road_df[['Latitude', 'Longitude']].values
            
            # Extract coordinates from predictions
            pred_coords = pred_df[['Latitude', 'Longitude']].values
            
            # Create BallTree for efficient nearest neighbor search
            # Convert to radians for haversine distance
            road_coords_rad = np.radians(road_coords)
            pred_coords_rad = np.radians(pred_coords)
            
            tree = BallTree(pred_coords_rad, metric='haversine')
            
            # Find closest prediction for each road point
            distances, indices = tree.query(road_coords_rad, k=1)
            
            # Convert distances from radians to meters (approximate)
            distances_meters = distances.flatten() * 6371000  # Earth radius in meters
            
            # Get predicted speeds for closest points
            closest_pred_speeds = pred_df.iloc[indices.flatten()]['predicted_speed'].values
            
            # Get real speeds for closest points (if available)
            if 'real_speed' in pred_df.columns:
                closest_real_speeds = pred_df.iloc[indices.flatten()]['real_speed'].values
                road_df['real_speed'] = closest_real_speeds
            else:
                road_df['real_speed'] = None
            
            # Add predicted speeds to road DataFrame
            road_df['predicted_speed'] = closest_pred_speeds
            road_df['prediction_distance_m'] = distances_meters
            
            # Use predicted speed as the main speed for visualization
            road_df['speed'] = road_df['predicted_speed']
            
            # Check for points that are too far from any prediction
            max_distance_threshold = 1000  # 1km threshold
            far_points = distances_meters > max_distance_threshold
            
            if far_points.any():
                print(f"Warning: {far_points.sum()} points in {road_name} {direction} are >{max_distance_threshold}m from predictions")
                # For points too far, use a default speed or interpolate
                road_df.loc[far_points, 'predicted_speed'] = road_df.loc[~far_points, 'predicted_speed'].mean()
                road_df.loc[far_points, 'speed'] = road_df.loc[~far_points, 'speed'].mean()
            
            print(f"Mapped predictions for {road_name} {direction}: "
                  f"{len(road_df)} points, avg distance: {distances_meters.mean():.1f}m")

    def get_prediction_statistics(self) -> dict:
        """
        Get statistics about the prediction mapping for all roads.
        
        Returns:
            Dictionary with statistics for each road
        """
        stats = {}
        
        for (road_name, direction), road_df in self.roads.items():
            if 'predicted_speed' not in road_df.columns:
                continue
                
            stats[(road_name, direction)] = {
                'total_points': len(road_df),
                'avg_predicted_speed': road_df['predicted_speed'].mean(),
                'min_predicted_speed': road_df['predicted_speed'].min(),
                'max_predicted_speed': road_df['predicted_speed'].max(),
                'avg_distance_to_prediction': road_df.get('prediction_distance_m', pd.Series([0])).mean(),
                'max_distance_to_prediction': road_df.get('prediction_distance_m', pd.Series([0])).max(),
                'points_with_predictions': road_df['predicted_speed'].notna().sum()
            }
        
        return stats

    def print_prediction_summary(self):
        """Print a summary of prediction statistics for all roads."""
        stats = self.get_prediction_statistics()
        
        if not stats:
            print("No prediction statistics available. Run apply_prediction_data() first.")
            return
        
        print("\n" + "="*80)
        print("PREDICTION MAPPING SUMMARY")
        print("="*80)
        
        for (road_name, direction), stat in stats.items():
            print(f"\n{road_name} {direction}:")
            print(f"  Points: {stat['points_with_predictions']}/{stat['total_points']}")
            print(f"  Speed: {stat['avg_predicted_speed']:.1f} mph (range: {stat['min_predicted_speed']:.1f}-{stat['max_predicted_speed']:.1f})")
            print(f"  Avg distance to prediction: {stat['avg_distance_to_prediction']:.1f}m")
            print(f"  Max distance to prediction: {stat['max_distance_to_prediction']:.1f}m")

    
    
    def draw_map(self):

        def get_color(speed, max_speed):
            if speed >= 0.85 * max_speed:
                return '#00FF00'  # Bright neon green
            elif speed >= 0.55 * max_speed:
                return '#FFA500'  # Bright orange
            else:
                return '#FF0000'  # Bright red
            
        center_lon = (self.bbox[0] + self.bbox[2]) / 2
        center_lat = (self.bbox[1] + self.bbox[3]) / 2

        m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='CartoDB dark_matter'
        )

        for (road_name, direction), df in self.roads.items():
            for i in range(len(df) - 1):
                lat1, lon1, speed1 = df.loc[i, ['Latitude', 'Longitude', 'speed']] # type: ignore
                lat2, lon2, speed2 = df.loc[i+1, ['Latitude', 'Longitude', 'speed']] # type: ignore
                raw_speed = df.loc[i, 'maxspeed']
                match = re.search(r'\d+', str(raw_speed))
                if match:
                    max_speed = float(match.group())
                else:
                    max_speed = 60

                dist = geodesic((lat1, lon1), (lat2, lon2)).meters
                if dist > DIST_THRESHOLD_METERS_MAX or dist < DIST_THRESHOLD_METERS_MIN:
                    continue  # Skip if too far or too close

                avg_speed = (speed1 + speed2) / 2
                color = get_color(avg_speed,max_speed)

                folium.PolyLine(
                    locations=[(lat1, lon1), (lat2, lon2)],
                    color=color,
                    weight=1,
                    opacity=0.9
                ).add_to(m)

        output_path = os.path.join(self.visualizations_path, "sorted_path.html")
        m.save(output_path)
        print("Saved map with distance filtering to 'sorted_path.html'")




    def draw_map_offset(self):

        def get_color(speed, max_speed):
            if speed >= 0.85 * max_speed:
                return '#00FF00'  # Neon green
            elif speed >= 0.55 * max_speed:
                return '#FFA500'  # Bright orange
            else:
                return '#FF0000'  # Bright red

        def get_maxspeed(raw_speed):
            match = re.search(r'\d+', str(raw_speed))
            return float(match.group()) if match else 60

        def apply_offset(lat, lon, bearing, direction):
            """Offset lat/lon a little perpendicular to bearing, based on direction."""
            offset_meters = -600 if direction.lower() in ["north", "east"] else 600

            # Convert bearing to radians and rotate 90°
            angle_rad = math.radians((bearing + 90) % 360)
            delta_lat = offset_meters * math.cos(angle_rad) / 111111
            delta_lon = offset_meters * math.sin(angle_rad) / (111111 * math.cos(math.radians(lat)))

            return lat + delta_lat, lon + delta_lon

        # Create dark base map
        center_lon = (self.bbox[0] + self.bbox[2]) / 2
        center_lat = (self.bbox[1] + self.bbox[3]) / 2

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='CartoDB dark_matter'
        )

        # Group by road name
        road_groups = {}
        for (road_name, direction), df in self.roads.items():
            road_groups.setdefault(road_name, {})[direction] = df

        for road_name, direction_map in road_groups.items():
            for direction, df in direction_map.items():
                for i in range(len(df) - 1):
                    lat1, lon1, speed1 = df.loc[i, ['Latitude', 'Longitude', 'speed']]
                    lat2, lon2, speed2 = df.loc[i + 1, ['Latitude', 'Longitude', 'speed']]
                    raw_speed = df.loc[i, 'maxspeed']
                    max_speed = get_maxspeed(raw_speed)
                    bearing = df.loc[i, 'bearing'] if 'bearing' in df.columns else 0

                    dist = geodesic((lat1, lon1), (lat2, lon2)).meters
                    if dist > DIST_THRESHOLD_METERS_MAX or dist < DIST_THRESHOLD_METERS_MIN:
                        continue

                    avg_speed = (speed1 + speed2) / 2
                    color = get_color(avg_speed, max_speed)

                    # Apply visual offset if road has both directions
                    has_opposite = len(direction_map) > 1
                    if has_opposite:
                        lat1, lon1 = apply_offset(lat1, lon1, bearing, direction)
                        lat2, lon2 = apply_offset(lat2, lon2, bearing, direction)

                    folium.PolyLine(
                        locations=[(lat1, lon1), (lat2, lon2)],
                        color=color,
                        weight=2,
                        opacity=0.95
                    ).add_to(m)

        output_path = os.path.join(self.visualizations_path, "direction_offset_map.html")
        m.save(output_path)
        print("✅ Saved map with directional offsets to 'direction_offset_map.html'")
        return m

    def draw_map_with_real_speed(self):
        """
        Draw map using real speed data instead of predicted speed.
        """
        def get_color(speed, max_speed):
            if speed >= 0.85 * max_speed:
                return '#00FF00'  # Neon green
            elif speed >= 0.55 * max_speed:
                return '#FFA500'  # Bright orange
            else:
                return '#FF0000'  # Bright red

        def get_maxspeed(raw_speed):
            match = re.search(r'\d+', str(raw_speed))
            return float(match.group()) if match else 60

        def apply_offset(lat, lon, bearing, direction):
            """Offset lat/lon a little perpendicular to bearing, based on direction."""
            offset_meters = -600 if direction.lower() in ["north", "east"] else 600

            # Convert bearing to radians and rotate 90°
            angle_rad = math.radians((bearing + 90) % 360)
            delta_lat = offset_meters * math.cos(angle_rad) / 111111
            delta_lon = offset_meters * math.sin(angle_rad) / (111111 * math.cos(math.radians(lat)))

            return lat + delta_lat, lon + delta_lon

        # Create dark base map
        center_lon = (self.bbox[0] + self.bbox[2]) / 2
        center_lat = (self.bbox[1] + self.bbox[3]) / 2

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            tiles='CartoDB dark_matter'
        )

        # Group by road name
        road_groups = {}
        for (road_name, direction), df in self.roads.items():
            road_groups.setdefault(road_name, {})[direction] = df

        for road_name, direction_map in road_groups.items():
            for direction, df in direction_map.items():
                for i in range(len(df) - 1):
                    lat1, lon1 = df.loc[i, ['Latitude', 'Longitude']]
                    lat2, lon2 = df.loc[i + 1, ['Latitude', 'Longitude']]
                    
                    # Use real speed if available, otherwise fall back to predicted speed
                    if 'real_speed' in df.columns and pd.notna(df.loc[i, 'real_speed']):
                        speed1 = df.loc[i, 'real_speed']
                        speed2 = df.loc[i + 1, 'real_speed'] if i + 1 < len(df) and pd.notna(df.loc[i + 1, 'real_speed']) else speed1
                    else:
                        speed1 = df.loc[i, 'speed']
                        speed2 = df.loc[i + 1, 'speed']
                    
                    raw_speed = df.loc[i, 'maxspeed']
                    max_speed = get_maxspeed(raw_speed)
                    bearing = df.loc[i, 'bearing'] if 'bearing' in df.columns else 0

                    dist = geodesic((lat1, lon1), (lat2, lon2)).meters
                    if dist > DIST_THRESHOLD_METERS_MAX or dist < DIST_THRESHOLD_METERS_MIN:
                        continue

                    avg_speed = (speed1 + speed2) / 2
                    color = get_color(avg_speed, max_speed)

                    # Apply visual offset if road has both directions
                    has_opposite = len(direction_map) > 1
                    if has_opposite:
                        lat1, lon1 = apply_offset(lat1, lon1, bearing, direction)
                        lat2, lon2 = apply_offset(lat2, lon2, bearing, direction)

                    folium.PolyLine(
                        locations=[(lat1, lon1), (lat2, lon2)],
                        color=color,
                        weight=2,
                        opacity=0.95
                    ).add_to(m)

        output_path = os.path.join(self.visualizations_path, "real_speed_map.html")
        m.save(output_path)
        print("✅ Saved map with real speed data to 'real_speed_map.html'")
        return m

    def draw_side_by_side_maps(self):
        """
        Create side-by-side maps showing both predicted and real speeds.
        Returns a tuple of (predicted_map, real_map) for use in Streamlit.
        """
        # Create predicted speed map
        predicted_map = self.draw_map_offset()
        
        # Create real speed map
        real_map = self.draw_map_with_real_speed()
        
        return predicted_map, real_map


"""
    mock_predictor = MockTrafficPredictor({
            'I 405 North': 'moderate',
            'I 405 South': 'free',
            'US 101 North': 'busy',
            'US 101 South': 'free',
            'I 5 North': 'busy',
            'I 5 South': 'free',
            'I 10 East': 'moderate',
            'I 10 West': 'moderate',
            'I 110 North': 'busy',
            'I 110 South': 'busy',
            'CA 110 North': 'busy',
            'CA 110 South': 'busy',
            'CA 170 North': 'moderate',
            'CA 170 South': 'free',
            'CA 118 East': 'free',
            'CA 118 West': 'free',
            'CA 134 East': 'moderate',
            'CA 134 West': 'free',
            'CA 2 North': 'moderate',
            'CA 2 South': 'moderate',
            'I 605 North': 'busy',
            'I 605': 'free',
            'I 210 East' : 'free',
            'I 210 West' : 'busy'
        })

        if predict_time is None:
            predict_time = datetime.now()

        for (road_name, direction), df in self.roads.items():
            #self.roads[(road_name, direction)] = add_weather_to_df(self.roads[(road_name, direction)], time = predict_time)
            print(f"Mocking for {road_name} - {direction}")
            df = mock_predictor.predict(df)
            print(df.head())
            df = sort_gps_by_greedy_path(df)
            self.roads[(road_name, direction)] = df

"""