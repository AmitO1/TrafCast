import pandas as pd
import glob
import os
import sys
import numpy as np
from sklearn.neighbors import BallTree
from datetime import datetime

EARTH_RADIUS_M = 6371000
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from roadmap.utils import add_weather_to_df

""""
handles this part in another file?
"""
DATA_DIR = os.path.join(PROJECT_ROOT, "data collection", "data", "I405(North) - March")
COORDINTE_DIR = os.path.join(PROJECT_ROOT,"data collection", "coordinates")
NETWORK_DIR = os.path.join(PROJECT_ROOT, "data", "Los Angeles", "coordinates")

test_data = os.path.join(DATA_DIR, "405_03*01*2025.xlsx")
test_coordinate = os.path.join(COORDINTE_DIR, "I 405 North.xlsx")
test_network = os.path.join(NETWORK_DIR, "I 405 North.csv")

df_coord = pd.read_excel(test_coordinate)
df_data = pd.read_excel(test_data)
df_network = pd.read_csv(test_network)
date = "2025-03-01"

def prepare_data_df(df_data: pd.DataFrame, coordinate: pd.DataFrame, date: str):
    """"
    first remove points with no observations, add date to the table and weather
    """
    df_data.drop(df_data[df_data["% Observed"] == 0.0].index, inplace=True)
    df_data["Time"] = pd.to_datetime(date + " " + df_data["Time"].astype(str),format="%Y-%m-%d %H:%M")

    df_data = add_coordinate(coordinate, df_data)

    #TODO send in in 1 hour intervals according to time column, send with time parameter
    df_data["Time_hour"] = df_data["Time"].dt.round("H")
    df_data = enrich_weather_hourly(df_data)

    return df_data

def add_coordinate(df_coord: pd.DataFrame, df_data: pd.DataFrame):
    df_coord = df_coord.sort_values(by="Abs PM").reset_index(drop=True)
    df_data = df_data.sort_values(by="Postmile (Abs)").reset_index(drop=True)

    coord_abs_pm = df_coord["Abs PM"].values
    coord_lat = df_coord["Latitude"].values
    coord_lon = df_coord["Longitude"].values

    def find_closest_index(target):
        return np.abs(coord_abs_pm - target).argmin()
    
    closest_indices = df_data["Postmile (Abs)"].apply(find_closest_index)

    df_data["Latitude"] = closest_indices.apply(lambda idx: coord_lat[idx]) # type: ignore
    df_data["Longitude"] = closest_indices.apply(lambda idx: coord_lon[idx]) # type: ignore
    
    return df_data

def enrich_weather_hourly(full_df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for t_hour, chunk in full_df.groupby("Time_hour", sort=False):
        enriched = add_weather_to_df(chunk.copy(),time=t_hour.to_pydatetime()) # type: ignore
        pieces.append(enriched)

    return pd.concat(pieces, ignore_index=True).sort_values("Time")

def build_sensor_index(data_df: pd.DataFrame) -> pd.DataFrame:
    sensors = (
        data_df
        .drop_duplicates(subset=["Latitude", "Longitude"])
        .reset_index(drop=True)
        .loc[:, ["Latitude", "Longitude"]]
    ).copy()

    sensors["sensor_id"] = np.arange(len(sensors), dtype=int)

    return sensors

def map_network_to_sensors(network_df: pd.DataFrame, sensors: pd.DataFrame, max_distance_m: float | None = None) -> pd.DataFrame:
    net = network_df.dropna(subset=["Latitude", "Longitude"]).copy()

    sensor_rad = np.radians(sensors[["Latitude", "Longitude"]].to_numpy())
    net_rad = np.radians(net[["Latitude", "Longitude"]].to_numpy())

    tree = BallTree(sensor_rad, metric="haversine")
    dist_rad, idx = tree.query(net_rad,k=1)
    dist_m = dist_rad[:, 0] * EARTH_RADIUS_M

    matched = net.copy()
    matched["sensor_id"] = sensors.iloc[idx[:, 0]].sensor_id.values
    matched["matched_sensor_lat"] = sensors.iloc[idx[:, 0]].Latitude.values
    matched["matched_sensor_lon"] = sensors.iloc[idx[:, 0]].Longitude.values
    matched["distance_m"] = dist_m

    if max_distance_m is not None:
        matched = matched[matched["distance_m"] <= max_distance_m].copy()
    
    return matched

def build_enriched_time_series(data_df: pd.DataFrame, sensor_map: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    sensor_lookup = sensors.set_index(["Latitude", "Longitude"])["sensor_id"]
    data_df = data_df.copy()
    data_df["sensor_id"] = sensor_lookup.loc[
        list(zip(data_df["Latitude"], data_df["Longitude"]))
    ].values

    enriched = (sensor_map[["sensor_id", "Latitude", "Longitude",
                            "lanes", "maxspeed", "ref", "direction", "road_name"]]
                .merge(data_df[["sensor_id", "Time", "AggSpeed", "% Observed", "weather"]],
                       on="sensor_id",
                       how="left"))

    return enriched


clean_data_df = prepare_data_df(df_data, df_coord,date)
sensors = build_sensor_index(clean_data_df)
network_mapped = map_network_to_sensors(df_network, sensors)
enriched = build_enriched_time_series(clean_data_df, network_mapped,sensors)
print(enriched)