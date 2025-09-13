# app.py
import streamlit as st
from datetime import datetime
from streamlit_folium import st_folium
from roadmap.RoadMap import RoadMapManager

SUPPORTED_CITIES = ["Los Angeles"]
LA_ROADS = [
    'I 405 North', 'I 405 South',
    'US 101 North', 'US 101 South',
    'I 5 North', 'I 5 South',
    'I 110 North', 'I 110 South',
    'CA 170 North', 'CA 170 South',
    'CA 118 East', 'CA 118 West',
    'CA 134 East', 'CA 134 West',
    'I 605 North', 'I 605 South',
    'I 210 East', 'I 210 West'
]
LA_BBOX = (-118.569946, 33.252470, -116.976929, 34.388779)

st.title("TrafCast: Traffic Forecasting for Los Angeles")

city = st.selectbox("Select City", SUPPORTED_CITIES)

@st.cache_resource
def get_map_manager(city_name):
    return RoadMapManager(city_name, LA_BBOX)

map_manager = get_map_manager(city)

selected_roads = st.multiselect("Select Roads to Load", LA_ROADS)

if selected_roads and st.button("Load Road Data"):
    with st.spinner("Loading road data..."):
        map_manager.set_roads(selected_roads)
        st.session_state["roads_loaded"] = True
    st.success("Road data loaded successfully.")

if st.session_state.get("roads_loaded"):
    default_date = st.session_state.get("selected_date", datetime.now().date())
    default_time = st.session_state.get("selected_time", datetime.now().time())

    st.date_input("Choose Date", value=default_date, key="selected_date")
    st.time_input("Choose Time", value=default_time, key="selected_time")

    predict_time = datetime.combine(
        st.session_state["selected_date"],
        st.session_state["selected_time"]
    )


    map_option = st.radio(
        "Choose map visualization:",
        ["Predicted Speed Only", "Real Speed Only", "Side by Side Comparison"],
        key="map_option"
    )
    
    if st.button("Apply Prediction"):
        with st.spinner("Running prediction and generating map..."):
            map_manager.apply_prediction_data(predict_time)
            
            if map_option == "Predicted Speed Only":
                folium_map = map_manager.draw_map_offset()
                st.session_state["folium_map"] = folium_map
                st.session_state["map_type"] = "predicted"
            elif map_option == "Real Speed Only":
                folium_map = map_manager.draw_map_with_real_speed()
                st.session_state["folium_map"] = folium_map
                st.session_state["map_type"] = "real"
            else:  # Side by Side Comparison
                predicted_map, real_map = map_manager.draw_side_by_side_maps()
                st.session_state["predicted_map"] = predicted_map
                st.session_state["real_map"] = real_map
                st.session_state["map_type"] = "side_by_side"
        st.success("Map updated!")

if st.session_state.get("map_type") == "side_by_side":
    if "predicted_map" in st.session_state and "real_map" in st.session_state:
        # Use container to control spacing
        with st.container():
            st.subheader("🟢 Predicted Speed")
            st_folium(
                st.session_state["predicted_map"],
                width=1200,
                height=600,
                returned_objects=[],
                key="predicted_map"
            )
        
        # Minimal spacing
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        
        with st.container():
            st.subheader("🔴 Real Speed")
            st_folium(
                st.session_state["real_map"],
                width=1200,
                height=600,
                returned_objects=[],
                key="real_map"
            )
            
elif "folium_map" in st.session_state:
    map_title = "Predicted Speed" if st.session_state.get("map_type") == "predicted" else "Real Speed"
    st.subheader(f"🗺️ {map_title}")
    st_folium(
        st.session_state["folium_map"],
        width=1000,
        height=1000,
        returned_objects=[],
        key="traffic_map"
    )
