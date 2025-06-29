from geopandas import GeoDataFrame
from networkx import MultiDiGraph
import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
from sklearn.neighbors import BallTree
import requests
from sklearn.cluster import KMeans
from datetime import datetime

def filter_by_direction(selected_road: GeoDataFrame, road_direction: str) -> GeoDataFrame:
    if road_direction == 'North':
        return selected_road[
            (selected_road['bearing'] >= 270) | (selected_road['bearing'] <= 90)
        ]
    elif road_direction == 'South':
        return selected_road[
            (selected_road['bearing'] > 90) & (selected_road['bearing'] < 270)
        ]
    elif road_direction == 'East':
        return selected_road[
            (selected_road['bearing'] >= 0) & (selected_road['bearing'] <= 180)
        ]
    elif road_direction == 'West':
        return selected_road[
            (selected_road['bearing'] > 180) & (selected_road['bearing'] < 360)
        ]
    else:
        raise ValueError(f"Invalid road_direction: {road_direction}. Must be one of: North, South, East, West.")

def add_weather_to_df(df: pd.DataFrame, num_clusters: int = 4 , api_key = 'FLMEW5QEEB8WT8YGUJXF6KAPK', time: datetime = datetime.now()) -> pd.DataFrame:
    if df.empty:
        df['weather'] = None
        return df
    
    coords = df[['Latitude', 'Longitude']].dropna().values
    kmeans = KMeans(n_clusters=min(num_clusters, len(coords)), random_state=42)
    df['weather_cluster'] = kmeans.fit_predict(coords)

    weather_data = {}
    date_str = time.strftime("%Y-%m-%d")

    for cluster_id in range(kmeans.n_clusters): # type: ignore
        lat, lon = kmeans.cluster_centers_[cluster_id]
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}"
        params = {
            "key": api_key,
            "unitGroup": "metric",
            "contentType": "json"
        }

        try:
            response = requests.get(url=url, params=params)
            response.raise_for_status()
            data = response.json()
            hours = data.get("days", [{}])[0].get("hours", [])

            def hour_diff(hour_entry):
                try:
                    return abs(datetime.strptime(hour_entry["datetime"], "%H:%M:%S") - datetime.strptime(target_hour, "%H:%M:%S"))
                except:
                    return datetime.max

            if hours:
                best_match = min(hours, key=hour_diff)
                weather = best_match.get("conditions", "Unknown")
                weather_time = best_match.get("datetime", None)
            else:
                weather = "Unknown"
                weather_time = None

        except Exception as e:
            print(f"Weather api error for cluster {cluster_id}: {e}")
            weather = "Unknown"
            weather_time = None
        
        weather_data[cluster_id] = {
            "conditions": weather,
            "datetime": weather_time
        }

    df['time'] = time
    df['weather'] = df['weather_cluster'].map(lambda x: weather_data[x]["conditions"])
    df['weather_time'] = df['weather_cluster'].map(lambda x: weather_data[x]["datetime"])
    df.drop(columns=['weather_cluster'], inplace=True)
    return df

def get_coordinates_from_network(G : MultiDiGraph, road_name: str, road_direction: str, add_weather_time: bool):

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    edges_motorway = edges[edges['highway'].isin(['motorway', 'motorway_link'])]

    selected_road = edges_motorway[
        edges_motorway['ref'].str.contains(road_name, na=False, case=False)
    ]

    selected_road = filter_by_direction(selected_road, road_direction)

    rows = []

    for _, row in selected_road.iterrows():
        lanes = row.get("lanes", None)
        maxspeed = row.get("maxspeed", None)
        road_name = row.get("name", None) # type: ignore
        ref = row.get("ref", None)
        geometry = row.geometry

        if isinstance(geometry, LineString):
            coords = geometry.coords
        elif isinstance(geometry, MultiLineString):
            coords = [pt for line in geometry.geoms for pt in line.coords]
        else:
            continue

        for lon, lat in coords:
            rows.append({
                "Longitude": lon,
                "Latitude": lat,
                "lanes": lanes,
                "maxspeed": maxspeed,
                "road_name": road_name,
                "ref": ref,
                "direction" : road_direction
            })

    # Step 6: Build DataFrame
    road_df = pd.DataFrame(rows)
    if add_weather_time:
        road_df = add_weather_to_df(road_df)

    print(f"Total points in {road_name} - {road_direction}: {len(road_df)}")
    return road_df


def sort_gps_by_greedy_path(df: pd.DataFrame) -> pd.DataFrame:
    """
    Greedy nearest-neighbor sorting of GPS coordinates.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Latitude' and 'Longitude' columns.
    
    Returns:
        pd.DataFrame: Reordered DataFrame.
    """
    coords_rad = np.radians(df[['Latitude', 'Longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')

    visited = np.zeros(len(df), dtype=bool)
    path = []
    current_idx = 0  # or use farthest-point-start logic

    for _ in range(len(df)):
        visited[current_idx] = True
        path.append(current_idx)

        dist, ind = tree.query([coords_rad[current_idx]], k=len(df))

        for next_idx in ind[0]:
            if not visited[next_idx]:
                current_idx = next_idx
                break

    return df.iloc[path].reset_index(drop=True)


def match_training_data_to_road_network(training_df, road_df):
    speed_coords = np.radians(training_df[['Latitude', 'Longitude']].values)
    points_coords = np.radians(road_df[['Latitude', 'Longitude']].values)

    tree = BallTree(speed_coords, metric='haversine')

    distances, indices = tree.query(points_coords, k=1)

    meters = distances[:, 0] * 6371000

    road_df['Speed'] = training_df.loc[indices[:, 0], 'Speed'].values

    road_df.to_csv("i405_geometry_points_with_speed.csv", index=False)