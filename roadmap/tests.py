import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
from RoadMap import RoadMapManager


la_roads = [
    'I 405 North', 'I 405 South',
    'US 101 North', 'US 101 South',
    'I 5 North', 'I 5 South',
    'I 10 East', 'I 10 West',
    'CA 110 North', 'CA 110 South',
    'CA 170 North', 'CA 170 South',
    'CA 118 East', 'CA 118 West',
    'CA 134 East', 'CA 134 West',
    'CA 2 North', 'CA 2 South'
]

lon_min = -118.569946  
lat_min =  33.252470   
lon_max = -116.976929  
lat_max =  34.388779
bbox = (lon_min, lat_min, lon_max, lat_max)
la_map = RoadMapManager('Los Angeles', bbox)
la_map.set_roads(la_roads, add_weather_time=False)
#la_map.get_roads()
la_map.apply_prediction_data()
la_map.draw_map()

exit(0)
