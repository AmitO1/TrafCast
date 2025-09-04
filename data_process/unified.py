import os
import sys
import pandas as pd
from process import prepare_data_df, build_enriched_time_series, build_sensor_index, map_network_to_sensors, normalize_lanes

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


DATA_DIR = os.path.join(PROJECT_ROOT, "data collection", "data", "I405(North) - March")
COORDINTE_DIR = os.path.join(PROJECT_ROOT,"data collection", "coordinates")
NETWORK_DIR = os.path.join(PROJECT_ROOT, "data", "Los Angeles", "coordinates")

test_coordinate = os.path.join(COORDINTE_DIR, "I 405 North.xlsx")
test_network = os.path.join(NETWORK_DIR, "I 405 North.csv")

df_coord = pd.read_excel(test_coordinate)
df_network = pd.read_csv(test_network)

full_df = pd.DataFrame() 

for i in range(1,32):
    if i <=9:
        test_data = os.path.join(DATA_DIR, f"405_03*0{i}*2025.xlsx")
        date = f"2025-03-0{i}"
    else:
        test_data = os.path.join(DATA_DIR, f"405_03*{i}*2025.xlsx")
        date = f"2025-03-{i}"

    df_data = pd.read_excel(test_data)
    clean_data_df = prepare_data_df(df_data, df_coord,date)
    sensors = build_sensor_index(clean_data_df)
    network_mapped = map_network_to_sensors(df_network, sensors)
    enriched = build_enriched_time_series(clean_data_df, network_mapped,sensors)
    full_df = pd.concat([full_df, enriched], ignore_index=True)
    print(f"finished {date}")

#enriched.drop('sensor_id', axis=1)
full_df['lanes'] = full_df['lanes'].apply(normalize_lanes)
full_df["maxspeed"] = full_df["maxspeed"].astype(str).str.extract(r"(\d+(?:\.\d+)?)").astype(float)
full_df.to_csv('exmaple.csv',index=False)
#print(enriched)



   

