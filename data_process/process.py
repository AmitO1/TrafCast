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

def compute_sensor_id(df: pd.DataFrame,
                      lat_col: str = "Latitude",
                      lon_col: str = "Longitude",
                      decimals: int = 6,
                      out_col: str = "sensor_id") -> pd.DataFrame:
    df[out_col] = (
        df[lat_col].round(decimals).astype(str)
        + ";" +
        df[lon_col].round(decimals).astype(str)
    )
    return df

def prepare_data_df(df_data: pd.DataFrame, coordinate: pd.DataFrame, date: str):
    """"
    first remove points with no observations, add date to the table and weather
    """
    df_data.drop(df_data[df_data["% Observed"] < 50].index, inplace=True)
    df_data["Time"] = pd.to_datetime(date + " " + df_data["Time"].astype(str),format="%Y-%m-%d %H:%M")

    df_data = add_coordinate(coordinate, df_data)
    df_data = compute_sensor_id(df_data)

    df_data["Time_hour"] = df_data["Time"].dt.round("h")
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
        .loc[:, ["Latitude", "Longitude"]]
        .copy()
        .reset_index(drop=True)
    )
    sensors = compute_sensor_id(sensors)  # adds 'sensor_id' as "lat;lon"
    # keep only what we need; you can also keep an integer 'sensor_idx' if you like
    return sensors[["sensor_id", "Latitude", "Longitude"]]

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

def build_enriched_time_series(data_df: pd.DataFrame,
                               sensor_map: pd.DataFrame,
                               sensors: pd.DataFrame) -> pd.DataFrame:
    # Ensure both sides have the same sensor_id key
    data_df = compute_sensor_id(data_df)  # from its own lat/lon

    enriched = (
        sensor_map[["sensor_id", "Latitude", "Longitude", "lanes", "maxspeed", "ref", "direction"]]
        .merge(
            data_df[["sensor_id", "Time", "AggSpeed", "% Observed", "weather"]],
            on="sensor_id",
            how="left"
        )
        .sort_values(["sensor_id", "Time"])
        .reset_index(drop=True)
    )
    return enriched


def normalize_lanes(value):
    if isinstance(value, list):
        try:
            return min(int(x) for x in value)
        except ValueError:
            return None
    try:
        return int(value)
    except ValueError:
        return None


