import os
import sys
import glob
import pandas as pd
from process import prepare_data_df, build_sensor_index, map_pms_to_sensors

index_map = {
    '405': 'I',
    '101': 'US',
    '101': 'US',
    '110': 'I',
    '170': 'CA',
    '118': 'CA',
    '134': 'CA',
    '605': 'I',
    '210': 'I',
    '5': 'I'
}
direction_map = {
    'E': 'East',
    'W': 'West',
    'N': 'North',
    'S': 'South'
}


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


DATA_DIR = os.path.join(PROJECT_ROOT, "data_collection", "data")
COORDINATE_DIR = os.path.join(PROJECT_ROOT,"data_collection", "coordinates")
full_df = pd.DataFrame() 

for entry in os.scandir(DATA_DIR):
    if entry.is_dir():
        road_number = entry.name
        road_name = index_map[road_number] +" "+ road_number 

        with os.scandir(entry.path) as it:
            for sub in it:
                if sub.is_dir() and sub.name in {'E', 'W', 'N', 'S'}:
                    direction = direction_map[sub.name]
                    print(f"Processing {road_name} {direction}")  
                    data_dir = os.path.join(entry.path, sub.name)
                    coordinate_dir = os.path.join(COORDINATE_DIR, f"{road_name} {direction}.xlsx")

                    for i in range(1,32):
                        if i <=9:
                            raw_data_pattern = os.path.join(data_dir, f"{road_number}_{sub.name}_03*0{i}*2025.xlsx")
                            date = f"2025-03-0{i}"
                        else:
                            raw_data_pattern = os.path.join(data_dir, f"{road_number}_{sub.name}_03*{i}*2025.xlsx")
                            date = f"2025-03-{i}"

                        matching_files = glob.glob(raw_data_pattern)
                        if not matching_files:
                            print(f"No data file found for {road_name} {direction} on {date}, skipping...")
                            continue
                        
                        raw_data = matching_files[0]

                        df_coord = pd.read_excel(coordinate_dir)
                        df_data = pd.read_excel(raw_data)
                        clean_data_df = prepare_data_df(df_data, df_coord,date)
                        sensors = build_sensor_index(clean_data_df)

                        enriched = map_pms_to_sensors(clean_data_df, sensors)
                        enriched["road_name"] = road_name
                        enriched["direction"] = direction

                        full_df = pd.concat([full_df, enriched], ignore_index=True)
                        print(f"finished {date}")
    

full_df.drop(columns=["Postmile (Abs)", "Postmile (CA)", "VDS", "Time_hour", "matched_sensor_lat", "matched_sensor_lon", "distance_m"], inplace=True)
desired_order = [
    "Time","sensor_id", "Latitude", "Longitude",
    "road_name", "direction", "# Lane Points",
    "% Observed", "weather", "Day", "AggSpeed"
]
full_df = full_df[desired_order]
full_df.rename(columns={
    "AggSpeed": "speed_mph",
    "# Lane Points": "lanes"
}, inplace=True)
full_df.to_csv('full_df_weather.csv',index=False)




   

