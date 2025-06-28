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
from utils import get_coordinates_from_network, sort_gps_by_greedy_path
from mock_predictor import MockTrafficPredictor
from geopy.distance import geodesic

DIST_THRESHOLD_METERS_MAX = 2000
DIST_THRESHOLD_METERS_MIN = 10

#TODO
#2. After matching need to append relevent data for predictions for exmaple only Weather, date, direction left:
#   Road: US-101 | Time: 17:45 Friday | Weather: Clear | Coordinates: (34.1, -118.3) | Lanes: 4 | Speed Limit: 65mph | Direction: North
#   keep in mind will need this also for training - maybe add another class for this handler
#
#3. Then predict using LLM - skip for now can be implmented later
#   LSTM - how many different roads can predict? predit all together? how many points from a given road in a time?

class RoadMapManager:

    def __init__(self, city: str,bbox: Tuple[float,float,float,float], base_data_dir: str = "data"):
        self.city = city
        self.bbox = bbox
        self.base_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", base_data_dir))
        self.city_path = os.path.join(self.base_data_dir, self.city)
        self.coordinates_path = os.path.join(self.city_path, 'coordinates') # could be use to save downloaded roads and then only change weather and time in df
        self.predictions_path = os.path.join(self.city_path, 'predictions')
        self.road_network_path = os.path.join(self.city_path, 'maps')
        self.visualizations_path = os.path.join(self.city_path, 'visualizations')

        self.roads = dict()

        self._validate_structure()
        self._load_road_network(self.bbox)

        
    def _validate_structure(self):
        for path in [self.coordinates_path, self.predictions_path, self.road_network_path, self.visualizations_path]:
            os.makedirs(path, exist_ok=True)
            
    @staticmethod
    def split_road_name_direction(road_name: str) -> Tuple[str, str]:
        parts = road_name.split()
        return " ".join(parts[:-1]), parts[-1]
    
    def set_roads(self, roads: List[str]):
        for road in roads:
            road_name, direction = self.split_road_name_direction(road)
            self.roads[(road_name,direction)] = get_coordinates_from_network(self.road_network, road_name, direction)
    
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
            
    def apply_prediction_data(self, predict_time = None):
        """
        Needed data to predict: 
        Gather data about weather in current day in time for each point
        Speed limit - from the road network
        Road name, Direction - key of the coordinates dict
        Cooridnate from the coordinate df
        Lanes - either from coordinate if given else from road network
        Time - input from user
        """
        mock_predictor = MockTrafficPredictor({
            'I 405': 'moderate'
        })
        for (road_name, direction), df in self.roads.items():
            print(f"Mocking for {road_name} - {direction}")
            df = mock_predictor.predict(df)
            print(df.head())
            df = sort_gps_by_greedy_path(df)
            self.roads[(road_name, direction)] = df

    def draw_map(self):

        def get_color(speed,max_speed):
            if speed >= 0.85 * max_speed:
                return 'green'
            elif speed >= 0.55 * max_speed:
                return 'orange'
            else:
                return 'red'
            
        center_lon = (self.bbox[0] + self.bbox[2]) / 2
        center_lat = (self.bbox[1] + self.bbox[3]) / 2

        m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13
        )

        for (road_name, direction), df in self.roads.items():
            for i in range(len(df) - 1):
                lat1, lon1, speed1 = df.loc[i, ['Latitude', 'Longitude', 'speed']] # type: ignore
                lat2, lon2, speed2 = df.loc[i+1, ['Latitude', 'Longitude', 'speed']] # type: ignore
                raw_speed = df.loc[i, 'maxspeed']
                max_speed = float(str(raw_speed).split()[0])

                dist = geodesic((lat1, lon1), (lat2, lon2)).meters
                if dist > DIST_THRESHOLD_METERS_MAX or dist < DIST_THRESHOLD_METERS_MIN:
                    continue  # Skip if too far or too close

                avg_speed = (speed1 + speed2) / 2
                color = get_color(avg_speed,max_speed)

                folium.PolyLine(
                    locations=[(lat1, lon1), (lat2, lon2)],
                    color=color,
                    weight=8,
                    opacity=0.9
                ).add_to(m)

        output_path = os.path.join(self.visualizations_path, "sorted_path.html")
        m.save(output_path)
        print("Saved map with distance filtering to 'sorted_path.html'")