import pandas as pd
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import folium

# Load coordinates CSV (with tab separator)
df = pd.read_csv('coordinate.csv', sep='\t')

# Print first rows to confirm
print(df.head())

# Compute bounding box (+ small margin)
lat_min = df['Latitude'].min() - 0.01
lat_max = df['Latitude'].max() + 0.01
lon_min = df['Longitude'].min() - 0.01
lon_max = df['Longitude'].max() + 0.01

bbox = (lon_min, lat_min, lon_max, lat_max)  # (left, bottom, right, top)

G = ox.graph_from_bbox(
    bbox=bbox,
    network_type='drive'
)


# Convert to GeoDataFrame of edges (roads)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Print columns to confirm
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

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Plot all roads (gray background)
edges.plot(ax=ax, linewidth=0.5, color='gray', alpha=0.5)

# Plot I-405 segments (in red)
if not selected_road.empty:
    selected_road.plot(ax=ax, linewidth=3, color='red', label='I-405 / San Diego Fwy')
else:
    print("⚠️ Couldn't find I-405 segments in this area.")

# Plot GPS points (blue dots)
ax.scatter(df['Longitude'], df['Latitude'], color='blue', s=10, label='GPS Points')

# Final touches
plt.title("I-405 Road around GPS Points")
plt.legend()
plt.show()



# test
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

# Create a Folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')  # You can also use 'Stamen Toner', 'CartoDB Positron', etc.

# Add I-405 road segments
for _, row in selected_road.iterrows():
    folium.PolyLine(locations=[(y, x) for x, y in row['geometry'].coords],
                    color='red', weight=5).add_to(m)

# Add GPS points
for _, row in df.iterrows():
    folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                        radius=3, color='blue', fill=True).add_to(m)

# Save map as HTML
m.save("traffic_map.html")
