# app.py
import streamlit as st
from datetime import datetime
from streamlit_folium import st_folium
from RoadMap import RoadMapManager

# Constants
SUPPORTED_CITIES = ["Los Angeles"]
LA_ROADS = [
    'I 405 North', 'I 405 South',
    'US 101 North', 'US 101 South',
    'I 5 North', 'I 5 South',
    'I 110 North', 'I 110 South',
    'CA 110 North', 'CA 110 South',
    'CA 170 North', 'CA 170 South',
    'CA 118 East', 'CA 118 West',
    'CA 134 East', 'CA 134 West',
    'I 605 North', 'I 605 South',
    'I 210 East', 'I 210 West'
]
LA_BBOX = (-118.569946, 33.252470, -116.976929, 34.388779)

# App title
st.title("TrafCast: Traffic Forecasting for Los Angeles")

# Select city (currently only one)
city = st.selectbox("Select City", SUPPORTED_CITIES)

# Initialize or get cached map manager
@st.cache_resource
def get_map_manager(city_name):
    return RoadMapManager(city_name, LA_BBOX)

map_manager = get_map_manager(city)

# Multiselect roads
selected_roads = st.multiselect("Select Roads to Load", LA_ROADS)

# Button to load road data (only once)
if selected_roads and st.button("Load Road Data"):
    with st.spinner("Loading road data..."):
        map_manager.set_roads(selected_roads)
        st.session_state["roads_loaded"] = True
    st.success("Road data loaded successfully.")

if st.session_state.get("roads_loaded"):
    # Safe defaults ONLY if not already set
    default_date = st.session_state.get("selected_date", datetime.now().date())
    default_time = st.session_state.get("selected_time", datetime.now().time())

    # Bind widgets to session state using key
    st.date_input("Choose Date", value=default_date, key="selected_date")
    st.time_input("Choose Time", value=default_time, key="selected_time")

    # Use selected values
    predict_time = datetime.combine(
        st.session_state["selected_date"],
        st.session_state["selected_time"]
    )


    if st.button("Apply Prediction"):
        with st.spinner("Running prediction and generating map..."):
            map_manager.apply_prediction_data(predict_time)
            folium_map = map_manager.draw_map_offset()
            st.session_state["folium_map"] = folium_map
        st.success("Map updated!")

# Show map if one exists
if "folium_map" in st.session_state:
    st_folium(
        st.session_state["folium_map"],
        width=1000,
        height=700,
        returned_objects=[],
        key="traffic_map"
    )
