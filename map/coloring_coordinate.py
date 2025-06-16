import pandas as pd
import osmnx as ox
import geopandas as gpd
from geopy.distance import geodesic
import shapely.geometry
from shapely.geometry import LineString, MultiLineString, Point
import shapely.ops
import folium
import os
import numpy as np
from sklearn.neighbors import BallTree

"""
this should be the first function used to get the data for the needed area
"""
# Load GPS points
df = pd.read_csv('Coordinate_Data_with_Speed.csv', sep=',')


if os.path.exists('/Users/amitomer/Desktop/לימודים/Deep Learning/TrafCast/map/road_network.graphml'):
    print("Map already exists")
    G = ox.load_graphml("road_network.graphml")

else:
    # Bounding box
    print("Downloading {addd map name}")
    lat_min = df['Latitude'].min() - 0.01
    lat_max = df['Latitude'].max() + 0.01
    lon_min = df['Longitude'].min() - 0.01
    lon_max = df['Longitude'].max() + 0.01

    bbox = (lon_min, lat_min, lon_max, lat_max)

    # Download road network
    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type='drive'
    )
    
    # Step 2: Add edge bearings
    G = ox.bearing.add_edge_bearings(G)
    ox.save_graphml(G, filepath="road_network.graphml")

"""# Add bearings to the edges
G = ox.bearing.add_edge_bearings(G)
print("test")"""

# Convert to edges GeoDataFrame
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Filter motorway + motorway_link
edges_motorway = edges[edges['highway'].isin(['motorway', 'motorway_link'])]

# Match I-405 segments (by name)
selected_road = edges_motorway[
    edges_motorway['ref'].str.contains("405", na=False, case=False)
]

#default >= 315 , <= 45
selected_road = selected_road[
    (selected_road['bearing'] >= 270) | (selected_road['bearing'] <= 90)
]

print("*******************************")
print(len(selected_road))
print("*******************************")

"""
second function, extract road points in order to plot the graph
"""
# Combine all selected road segments into a single LineString
i405_geometry = selected_road.unary_union


# Prepare list of all coordinates (lat/lon points) from i405_geometry
points = []

if isinstance(i405_geometry, MultiLineString):
    for line in i405_geometry.geoms:
        points.extend(list(line.coords))
elif isinstance(i405_geometry, LineString):
    points.extend(list(i405_geometry.coords))
else:
    print("Unexpected geometry type:", type(i405_geometry))

"""
save points as csv not sure if really needed but for me for checking
"""
# Convert to DataFrame with columns: Longitude, Latitude
df_i405_points = pd.DataFrame(points, columns=["Longitude", "Latitude"])

# Save to CSV
df_i405_points.to_csv("i405_geometry_points.csv", index=False)

print("✅ Saved points to i405_geometry_points.csv")

"""
match each point from the downloaded geomatry to a point from the given csv
"""

#base find closeset point
"""# Function to find the closest speed point to a given point
def find_closest_speed(lat, lon, speed_df):
    min_dist = float('inf')
    closest_speed = None
    for _, row in speed_df.iterrows():
        gps_point = (row['Latitude'], row['Longitude'])
        dist = geodesic((lat, lon), gps_point).meters
        if dist < min_dist:
            min_dist = dist
            closest_speed = row['Speed']
    return closest_speed

# Apply the function to each road point
df_i405_points['Speed'] = df_i405_points.apply(
    lambda row: find_closest_speed(row['Latitude'], row['Longitude'], df), axis=1
)"""

# Convert lat/lon to radians
speed_coords = np.radians(df[['Latitude', 'Longitude']].values)
points_coords = np.radians(df_i405_points[['Latitude', 'Longitude']].values)

# Create a BallTree using haversine metric (distance on a sphere)
tree = BallTree(speed_coords, metric='haversine')

# Query the closest point in the speed dataset for each road point
distances, indices = tree.query(points_coords, k=1)

# Convert distances from radians to meters (Earth radius ≈ 6371000 m)
meters = distances[:, 0] * 6371000

# Fetch the closest speeds using indices
df_i405_points['Speed'] = df.loc[indices[:, 0], 'Speed'].values

# Save the result
df_i405_points.to_csv("i405_geometry_points_with_speed.csv", index=False)


"""
final function to plot the map
"""
# Load your data
df = pd.read_csv("i405_geometry_points_with_speed.csv")

# Convert to radians for Haversine
coords_rad = np.radians(df[['Latitude', 'Longitude']].values)

# Build BallTree for fast nearest neighbor search
tree = BallTree(coords_rad, metric='haversine')

# Initialize
visited = np.zeros(len(df), dtype=bool)
path = []
current_idx = 0  # Start at first point (you can choose any)

for _ in range(len(df)):
    visited[current_idx] = True
    path.append(current_idx)

    # Query neighbors of the current point
    dist, ind = tree.query([coords_rad[current_idx]], k=len(df))

    # Find the closest unvisited neighbor
    for next_idx in ind[0]:
        if not visited[next_idx]:
            current_idx = next_idx
            break

# Reorder df by the found path
df = df.iloc[path].reset_index(drop=True)

# Map center
avg_lat = df['Latitude'].mean()
avg_lon = df['Longitude'].mean()

m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13)

# Define speed color
def get_color(speed):
    if speed > 60:
        return 'green'
    elif speed > 20:
        return 'orange'
    else:
        return 'red'

# Threshold (in meters)
DIST_THRESHOLD_METERS_MAX = 2000  # adjust as needed
DIST_THRESHOLD_METERS_MIN = 15

# Draw lines only between close points
for i in range(len(df) - 1):
    lat1, lon1, speed1 = df.loc[i, ['Latitude', 'Longitude', 'Speed']]
    lat2, lon2, speed2 = df.loc[i+1, ['Latitude', 'Longitude', 'Speed']]

    dist = geodesic((lat1, lon1), (lat2, lon2)).meters
    if dist > DIST_THRESHOLD_METERS_MAX or dist < DIST_THRESHOLD_METERS_MIN:
        continue  # Skip if too far / small

    avg_speed = (speed1 + speed2) / 2
    color = get_color(avg_speed)

    folium.PolyLine(
        locations=[(lat1, lon1), (lat2, lon2)],
        color=color,
        weight=4,
        opacity=0.9
    ).add_to(m)


# Save
m.save("i405_colored_segments_filtered.html")
print("✅ Saved filtered colored map: i405_colored_segments_filtered.html")

