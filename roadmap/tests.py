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


    # Map visualization options
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
        
        # Show prediction statistics
        if hasattr(map_manager, 'get_prediction_statistics'):
            stats = map_manager.get_prediction_statistics()
            if stats:
                st.info("📊 **Prediction Statistics:**")
                for (road_name, direction), stat in list(stats.items())[:3]:  # Show first 3 roads
                    st.write(f"**{road_name} {direction}:** {stat['points_with_predictions']}/{stat['total_points']} points, "
                            f"Avg speed: {stat['avg_predicted_speed']:.1f} mph")

# Show map(s) based on visualization type
if st.session_state.get("map_type") == "side_by_side":
    # Side by side comparison
    if "predicted_map" in st.session_state and "real_map" in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Predicted Speed")
            st_folium(
                st.session_state["predicted_map"],
                width=500,
                height=500,
                returned_objects=[],
                key="predicted_map"
            )
        
        with col2:
            st.subheader("🔴 Real Speed")
            st_folium(
                st.session_state["real_map"],
                width=500,
                height=500,
                returned_objects=[],
                key="real_map"
            )
            
elif "folium_map" in st.session_state:
    # Single map display
    map_title = "Predicted Speed" if st.session_state.get("map_type") == "predicted" else "Real Speed"
    st.subheader(f"🗺️ {map_title}")
    st_folium(
        st.session_state["folium_map"],
        width=1000,
        height=700,
        returned_objects=[],
        key="traffic_map"
    )
