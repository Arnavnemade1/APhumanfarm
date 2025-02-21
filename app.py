import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from utils.data_helpers import load_crop_data, load_farm_records, save_record
from utils.visualization import create_gauge_chart, plot_history

# Page configuration
st.set_page_config(
    page_title="FarmTracker",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {color:#2E7D32; font-size:42px; font-weight:bold; text-align:center;}
    .sub-header {font-size:30px; font-weight:bold; color:#388E3C;}
    .card {background-color: #f0f2f6; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
    .status-optimal {color: #2E7D32; font-weight: bold;}
    .status-warning {color: #FF9800; font-weight: bold;}
    .status-danger {color: #D32F2F; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Initialize session state for tracking entered data
if "records" not in st.session_state:
    st.session_state.records = []
    
# Title and description
st.markdown('<p class="main-header">FarmTracker 🌱</p>', unsafe_allow_html=True)
st.markdown("Track optimal growing conditions for your fruits and vegetables")

# Sidebar for navigation
st.sidebar.image("https://www.svgrepo.com/show/484988/leaf.svg", width=100)
page = st.sidebar.radio("Navigation", ["Dashboard", "Add Measurements", "Crop Database", "Historical Data"])

# Load the crop database
crops_df = load_crop_data()
crop_list = crops_df.index.tolist()

if page == "Dashboard":
    st.markdown('<p class="sub-header">Current Farm Conditions</p>', unsafe_allow_html=True)
    
    # Current date and time
    current_time = datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
    st.markdown(f"**Last Updated:** {current_time}")
    
    # Select crop to monitor
    selected_crop = st.selectbox("Select crop to monitor", crop_list)
    
    # Get optimal conditions for selected crop
    crop_data = crops_df.loc[selected_crop]
    
    # Display dashboard in three columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Temperature (°C)")
        
        # Simulated current value (would be sensor data in production)
        current_temp = st.slider("Current temperature", 0, 40, 24)
        
        # Check if within optimal range
        if crop_data["temp_min"] <= current_temp <= crop_data["temp_max"]:
            status = "optimal"
        elif abs(current_temp - crop_data["temp_min"]) <= 3 or abs(current_temp - crop_data["temp_max"]) <= 3:
            status = "warning"
        else:
            status = "danger"
            
        st.markdown(f'<p class="status-{status}">Status: {status.upper()}</p>', unsafe_allow_html=True)
        st.markdown(f"Optimal range: {crop_data['temp_min']} - {crop_data['temp_max']} °C")
        
        # Display gauge chart
        fig = create_gauge_chart(
            current_value=current_temp,
            min_value=0,
            max_value=40,
            optimal_min=crop_data["temp_min"],
            optimal_max=crop_data["temp_max"],
            title="Temperature"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Humidity (%)")
        
        # Simulated current value
        current_humidity = st.slider("Current humidity", 0, 100, 70)
        
        # Check if within optimal range
        if crop_data["humidity_min"] <= current_humidity <= crop_data["humidity_max"]:
            status = "optimal"
        elif abs(current_humidity - crop_data["humidity_min"]) <= 5 or abs(current_humidity - crop_data["humidity_max"]) <= 5:
            status = "warning"
        else:
            status = "danger"
            
        st.markdown(f'<p class="status-{status}">Status: {status.upper()}</p>', unsafe_allow_html=True)
        st.markdown(f"Optimal range: {crop_data['humidity_min']} - {crop_data['humidity_max']}%")
        
        # Display gauge chart
        fig = create_gauge_chart(
            current_value=current_humidity,
            min_value=0,
            max_value=100,
            optimal_min=crop_data["humidity_min"],
            optimal_max=crop_data["humidity_max"],
            title="Humidity"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Soil pH")
        
        # Simulated current value
        current_ph = st.slider("Current pH", 4.0, 9.0, 6.5, 0.1)
        
        # Check if within optimal range
        if crop_data["ph_min"] <= current_ph <= crop_data["ph_max"]:
            status = "optimal"
        elif abs(current_ph - crop_data["ph_min"]) <= 0.5 or abs(current_ph - crop_data["ph_max"]) <= 0.5:
            status = "warning"
        else:
            status = "danger"
            
        st.markdown(f'<p class="status-{status}">Status: {status.upper()}</p>', unsafe_allow_html=True)
        st.markdown(f"Optimal range: {crop_data['ph_min']} - {crop_data['ph_max']}")
        
        # Display gauge chart
        fig = create_gauge_chart(
            current_value=current_ph,
            min_value=4.0,
            max_value=9.0,
            optimal_min=crop_data["ph_min"],
            optimal_max=crop_data["ph_max"],
            title="pH"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations based on current conditions
    st.subheader("Recommendations")
    
    recommendations = []
    
    if current_temp < crop_data["temp_min"]:
        recommendations.append(f"🔥 Temperature is too low. Consider greenhouse heating or protective covers.")
    elif current_temp > crop_data["temp_max"]:
        recommendations.append(f"❄️ Temperature is too high. Consider shade cloth or increased ventilation.")
        
    if current_humidity < crop_data["humidity_min"]:
        recommendations.append(f"💧 Humidity is too low. Consider misting or increasing irrigation.")
    elif current_humidity > crop_data["humidity_max"]:
        recommendations.append(f"☀️ Humidity is too high. Improve ventilation and reduce overhead watering.")
        
    if current_ph < crop_data["ph_min"]:
        recommendations.append(f"🧪 pH is too acidic. Consider adding agricultural lime.")
    elif current_ph > crop_data["ph_max"]:
        recommendations.append(f"🧪 pH is too alkaline. Consider adding sulfur or acidic organic matter.")
        
    if not recommendations:
        st.success("All conditions are optimal for growing " + selected_crop)
    else:
        for rec in recommendations:
            st.warning(rec)

elif page == "Add Measurements":
    st.markdown('<p class="sub-header">Record Farm Measurements</p>', unsafe_allow_html=True)
    
    # Form for entering new measurements
    with st.form("measurement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.date.today())
            time = st.time_input("Time", datetime.datetime.now().time())
            location = st.text_input("Field/Location", "Main Field")
            crop = st.selectbox("Crop", crop_list)
            
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=70)
            soil_ph = st.number_input("Soil pH", min_value=4.0, max_value=9.0, value=6.5, step=0.1)
            soil_moisture = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=60)
            
        notes = st.text_area("Notes", "")
        submitted = st.form_submit_button("Save Measurement")
        
        if submitted:
            # Create record
            record = {
                "date": date.strftime("%Y-%m-%d"),
                "time": time.strftime("%H:%M"),
                "location": location,
                "crop": crop,
                "temperature": temperature,
                "humidity": humidity,
                "soil_ph": soil_ph,
                "soil_moisture": soil_moisture,
                "notes": notes
            }
            
            # Save to session state (would save to database in production)
            st.session_state.records.append(record)
            save_record(record)
            
            st.success("Measurement recorded successfully!")
            
    # Display recent measurements
    if st.session_state.records:
        st.subheader("Recent Measurements")
        recent_records = pd.DataFrame(st.session_state.records[-5:])
        st.table(recent_records[["date", "time", "location", "crop", "temperature", "humidity", "soil_ph"]])

elif page == "Crop Database":
    st.markdown('<p class="sub-header">Crop Growing Conditions Database</p>', unsafe_allow_html=True)
    
    # Filter options
    filter_col1, filter_col2 = st.columns([1, 2])
    
    with filter_col1:
        search = st.text_input("Search crops", "")
        
    # Display crop database
    if search:
        filtered_df = crops_df[crops_df.index.str.contains(search, case=False)]
    else:
        filtered_df = crops_df
        
    # Add display columns for better readability
    display_df = filtered_df.copy()
    display_df["Temperature Range (°C)"] = display_df.apply(lambda x: f"{x['temp_min']} - {x['temp_max']}", axis=1)
    display_df["Humidity Range (%)"] = display_df.apply(lambda x: f"{x['humidity_min']} - {x['humidity_max']}", axis=1)
    display_df["pH Range"] = display_df.apply(lambda x: f"{x['ph_min']} - {x['ph_max']}", axis=1)
    
    # Display the simplified table
    st.dataframe(display_df[["Temperature Range (°C)", "Humidity Range (%)", "pH Range", "water_needs", "growing_season", "days_to_harvest"]])
    
    # Detailed crop information
    st.subheader("Detailed Crop Information")
    selected_crop = st.selectbox("Select crop for detailed information", filtered_df.index.tolist())
    
    crop_data = crops_df.loc[selected_crop]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {selected_crop}")
        st.markdown(f"**Growing Season:** {crop_data['growing_season']}")
        st.markdown(f"**Days to Harvest:** {crop_data['days_to_harvest']}")
        st.markdown(f"**Water Needs:** {crop_data['water_needs'].title()}")
        st.markdown(f"**Planting Depth:** {crop_data['planting_depth']} cm")
        st.markdown(f"**Row Spacing:** {crop_data['row_spacing']} cm")
        st.markdown(f"**Plant Spacing:** {crop_data['plant_spacing']} cm")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        # Create comparative charts
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        
        # Temperature range
        temp_range = np.arange(0, 41)
        temp_suitability = [(1 if crop_data['temp_min'] <= t <= crop_data['temp_max'] else 
                            0.5 if min(abs(t - crop_data['temp_min']), abs(t - crop_data['temp_max'])) <= 3 else 0) 
                           for t in temp_range]
        
        ax[0].fill_between(temp_range, temp_suitability, alpha=0.3, color='green')
        ax[0].set_xlim(0, 40)
        ax[0].set_ylim(0, 1.1)
        ax[0].set_title(f"Optimal Temperature Range: {crop_data['temp_min']}°C - {crop_data['temp_max']}°C")
        ax[0].set_xlabel("Temperature (°C)")
        ax[0].set_ylabel("Suitability")
        
        # Humidity range
        humidity_range = np.arange(0, 101)
        humidity_suitability = [(1 if crop_data['humidity_min'] <= h <= crop_data['humidity_max'] else 
                                0.5 if min(abs(h - crop_data['humidity_min']), abs(h - crop_data['humidity_max'])) <= 5 else 0) 
                               for h in humidity_range]
        
        ax[1].fill_between(humidity_range, humidity_suitability, alpha=0.3, color='blue')
        ax[1].set_xlim(0, 100)
        ax[1].set_ylim(0, 1.1)
        ax[1].set_title(f"Optimal Humidity Range: {crop_data['humidity_min']}% - {crop_data['humidity_max']}%")
        ax[1].set_xlabel("Humidity (%)")
        ax[1].set_ylabel("Suitability")
        
        # pH range
        ph_range = np.arange(4.0, 9.1, 0.1)
        ph_suitability = [(1 if crop_data['ph_min'] <= p <= crop_data['ph_max'] else 
                          0.5 if min(abs(p - crop_data['ph_min']), abs(p - crop_data['ph_max'])) <= 0.5 else 0) 
                         for p in ph_range]
        
        ax[2].fill_between(ph_range, ph_suitability, alpha=0.3, color='purple')
        ax[2].set_xlim(4, 9)
        ax[2].set_ylim(0, 1.1)
        ax[2].set_title(f"Optimal pH Range: {crop_data['ph_min']} - {crop_data['ph_max']}")
        ax[2].set_xlabel("pH")
        ax[2].set_ylabel("Suitability")
        
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Historical Data":
    st.markdown('<p class="sub-header">Historical Data Analysis</p>', unsafe_allow_html=True)
    
    # Load historical data (would be from database in production)
    historical_data = load_farm_records()
    
    if historical_data.empty:
        st.info("No historical data available yet. Start recording measurements to see analysis here.")
    else:
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input(
                "Date range",
                value=(
                    datetime.datetime.strptime(historical_data["date"].min(), "%Y-%m-%d").date(),
                    datetime.datetime.strptime(historical_data["date"].max(), "%Y-%m-%d").date()
                ),
                key="date_range"
            )
            
        with col2:
            locations = historical_data["location"].unique().tolist()
            selected_location = st.multiselect("Select locations", locations, default=locations[0] if locations else None)
            
        with col3:
            crops = historical_data["crop"].unique().tolist()
            selected_crop = st.multiselect("Select crops", crops, default=crops[0] if crops else None)
            
        # Filter data
        filtered_data = historical_data.copy()
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_data = filtered_data[(filtered_data["date"] >= start_date.strftime("%Y-%m-%d")) & 
                                        (filtered_data["date"] <= end_date.strftime("%Y-%m-%d"))]
            
        if selected_location:
            filtered_data = filtered_data[filtered_data["location"].isin(selected_location)]
            
        if selected_crop:
            filtered_data = filtered_data[filtered_data["crop"].isin(selected_crop)]
            
        # Convert date to datetime for better plotting
        filtered_data["datetime"] = pd.to_datetime(filtered_data["date"] + " " + filtered_data["time"])
        
        # Display trends
        st.subheader("Environmental Trends")
        
        # Create tabs for different metrics
        tab1, tab2, tab3 = st.tabs(["Temperature", "Humidity", "Soil pH"])
        
        with tab1:
            # Temperature trends
            fig = px.line(
                filtered_data,
                x="datetime",
                y="temperature",
                color="location",
                markers=True,
                title="Temperature Trends Over Time",
                labels={"temperature": "Temperature (°C)", "datetime": "Date/Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add crop optimal ranges
            if selected_crop and len(selected_crop) == 1:
                crop_data = crops_df.loc[selected_crop[0]]
                fig.add_hline(y=crop_data["temp_min"], line_dash="dash", line_color="green", 
                             annotation_text=f"Min ({crop_data['temp_min']}°C)")
                fig.add_hline(y=crop_data["temp_max"], line_dash="dash", line_color="red",
                             annotation_text=f"Max ({crop_data['temp_max']}°C)")
                st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Humidity trends
            fig = px.line(
                filtered_data,
                x="datetime",
                y="humidity",
                color="location",
                markers=True,
                title="Humidity Trends Over Time",
                labels={"humidity": "Humidity (%)", "datetime": "Date/Time"}
            )
            
            # Add crop optimal ranges
            if selected_crop and len(selected_crop) == 1:
                crop_data = crops_df.loc[selected_crop[0]]
                fig.add_hline(y=crop_data["humidity_min"], line_dash="dash", line_color="green",
                             annotation_text=f"Min ({crop_data['humidity_min']}%)")
                fig.add_hline(y=crop_data["humidity_max"], line_dash="dash", line_color="red",
                             annotation_text=f"Max ({crop_data['humidity_max']}%)")
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            # Soil pH trends
            fig = px.line(
                filtered_data,
                x="datetime",
                y="soil_ph",
                color="location",
                markers=True,
                title="Soil pH Trends Over Time",
                labels={"soil_ph": "Soil pH", "datetime": "Date/Time"}
            )
            
            # Add crop optimal ranges
            if selected_crop and len(selected_crop) == 1:
                crop_data = crops_df.loc[selected_crop[0]]
                fig.add_hline(y=crop_data["ph_min"], line_dash="dash", line_color="green",
                             annotation_text=f"Min ({crop_data['ph_min']})")
                fig.add_hline(y=crop_data["ph_max"], line_dash="dash", line_color="red",
                             annotation_text=f"Max ({crop_data['ph_max']})")
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Statistical summary
        st.subheader("Statistical Summary")
        summary_stats = filtered_data.groupby("location")[["temperature", "humidity", "soil_ph", "soil_moisture"]].describe()
        st.dataframe(summary_stats)
        
        # Download data option
        st.download_button(
            label="Download Filtered Data as CSV",
            data=filtered_data.to_csv(index=False).encode('utf-8'),
            file_name=f'farm_data_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
