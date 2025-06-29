import pandas as pd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString, MultiLineString
from RoadMap import RoadMapManager


la_roads = [
    'I 405 North'
]

lon_min = -118.569946  
lat_min =  33.252470   
lon_max = -116.976929  
lat_max =  34.388779
bbox = (lon_min, lat_min, lon_max, lat_max)
la_map = RoadMapManager('Los Angeles', bbox)
la_map.set_roads(la_roads)
#la_map.get_roads()
la_map.apply_prediction_data()
la_map.draw_map()

exit(0)
