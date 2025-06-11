import folium
import pandas as pd
import numpy as np

# Read coordinates from CSV file
df = pd.read_csv('coordinate.csv', sep='\t')

# Calculate the center point for the map
center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

# Create a base map centered on the average of all camera locations
m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

# Sort the points by longitude to ensure proper ordering
df = df.sort_values('Longitude')

# Create interpolated points
num_points = 100  # Number of points for smooth line
x = np.linspace(df['Longitude'].min(), df['Longitude'].max(), num_points)
y = np.interp(x, df['Longitude'], df['Latitude'])

# Create coordinates for the smooth line
smooth_coords = list(zip(y, x))

# Draw the smooth line
folium.PolyLine(
    smooth_coords,
    color='red',
    weight=3,
    opacity=0.8
).add_to(m)

# Add markers for the camera locations
for lat, lon in zip(df['Latitude'], df['Longitude']):
    folium.CircleMarker(
        location=(lat, lon),
        radius=4,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        popup=f'Camera Location: {lat:.4f}, {lon:.4f}'
    ).add_to(m)

# Save map to HTML
m.save("i405_map.html")
