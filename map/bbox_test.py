import pandas as pd
import folium

# Load GPS data
df = pd.read_csv('Coordinate_Data_with_Speed.csv', sep=',')

# Compute bounding box with margin
lat_min = df['Latitude'].min() - 0.01
lat_max = df['Latitude'].max() + 0.01
lon_min = df['Longitude'].min() - 0.01
lon_max = df['Longitude'].max() + 0.01

# Define bounding box corner points (label, lat, lon)
corners = [
    ("Top Left", lat_max, lon_min),
    ("Top Right", lat_max, lon_max),
    ("Bottom Left", lat_min, lon_min),
    ("Bottom Right", lat_min, lon_max)
]

# Compute center of the bounding box
center_lat = (lat_max + lat_min) / 2
center_lon = (lon_max + lon_min) / 2

# Create a Folium map centered at the bbox center
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# Add markers for the corners
for label, lat, lon in corners:
    folium.Marker(
        location=[lat, lon],
        popup=label,
        icon=folium.Icon(color='blue')
    ).add_to(m)

# Draw the bounding box rectangle
folium.PolyLine(
    locations=[
        (lat_max, lon_min),
        (lat_max, lon_max),
        (lat_min, lon_max),
        (lat_min, lon_min),
        (lat_max, lon_min)  # close the loop
    ],
    color="red",
    weight=2,
    dash_array="5"
).add_to(m)

# Save the map
m.save("bbox_debug_map.html")
print("✅ Saved bounding box debug map as bbox_debug_map.html")
