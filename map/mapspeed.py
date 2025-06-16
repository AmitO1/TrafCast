import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import folium

# Load coordinates CSV
df = pd.read_csv('Coordinate_Data_with_Speed.csv', sep=',')

# Print first rows to confirm
print(df.head())

# Compute bounding box (+ small margin)
lat_min = df['Latitude'].min() - 0.01
lat_max = df['Latitude'].max() + 0.01
lon_min = df['Longitude'].min() - 0.01
lon_max = df['Longitude'].max() + 0.01

bbox = (lon_min, lat_min, lon_max, lat_max)  # (left, bottom, right, top)

# Download road network
G = ox.graph_from_bbox(
    bbox=bbox,
    network_type='drive'
)

# Convert to GeoDataFrame of edges (roads)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

print("Available columns:", edges.columns)

# Filter only 'motorway' class roads (freeways)
edges_motorway = edges[edges['highway'] == 'motorway']

# Match I-405 segments using name field
selected_road = edges_motorway[
    edges_motorway['name'].str.contains("405|San Diego|Interstate", na=False, case=False)
]

# Show unique matching names
print("Matching road names:")
print(selected_road[['name']].drop_duplicates())

# Snap each GPS point to nearest edge (returns (u, v, key))
nearest_edges = ox.distance.nearest_edges(G, df['Longitude'].values, df['Latitude'].values)

# Add snapped edge to df
df['nearest_edge'] = nearest_edges

# Group speeds by edge → average speed per edge
edge_speeds = df.groupby('nearest_edge')['Speed'].mean().to_dict()

# Function to choose color based on speed
def get_color(speed):
    if speed > 60:
        return 'green'
    elif speed > 40:
        return 'orange'
    else:
        return 'red'

# Center map on GPS points
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

# Create Folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

# Add I-405 segments with color according to speed
for _, row in selected_road.iterrows():
    u, v, key = row.name
    edge_id = (u, v, key)

    speed = edge_speeds.get(edge_id, None)
    
    if speed is not None:
        color = get_color(speed)
    else:
        color = 'gray'  # no data
    
    # Plot segment
    if row['geometry'].geom_type == 'LineString':
        coords = [(y, x) for x, y in row['geometry'].coords]
        folium.PolyLine(locations=coords, color=color, weight=5).add_to(m)
    elif row['geometry'].geom_type == 'MultiLineString':
        for line in row['geometry']:
            coords = [(y, x) for x, y in line.coords]
            folium.PolyLine(locations=coords, color=color, weight=5).add_to(m)

# Add GPS points (blue)
for _, row in df.iterrows():
    folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                        radius=3, color='blue', fill=True).add_to(m)

# Save map
m.save("traffic_map_colored_by_speed.html")

print("✅ Map saved as traffic_map_colored_by_speed.html")
