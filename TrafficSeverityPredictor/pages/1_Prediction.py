"""
Traffic Severity Prediction App - Prediction Page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.model import initialize_model, predict_severity

# Set page title
st.set_page_config(
    page_title="Make a Prediction - Traffic Severity Prediction",
    page_icon="🚦",
    layout="wide"
)

# Initialize model if not already in session state
if 'model' not in st.session_state:
    st.session_state.model = initialize_model()
    
# Initialize prediction history if not already in session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame()

# Page title
st.title("🔮 Predict Traffic Incident Severity")
st.markdown("Enter the details below to predict the severity of a traffic incident.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    # Input parameters - left column
    day = st.selectbox(
        "Day of Week",
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        index=0
    )
    
    hour = st.slider(
        "Hour of Day (24-hour)",
        min_value=0,
        max_value=23,
        value=8,
        step=1
    )
    
    weather = st.selectbox(
        "Weather Condition",
        options=['Clear', 'Rain', 'Snow', 'Fog', 'Windy'],
        index=0
    )
    
    road_type = st.selectbox(
        "Road Type",
        options=['Highway', 'Urban', 'Residential', 'Rural'],
        index=0
    )

with col2:
    # Input parameters - right column
    junction_type = st.selectbox(
        "Junction Type",
        options=['Intersection', 'Roundabout', 'T-Junction', 'None'],
        index=0
    )
    
    traffic_volume = st.slider(
        "Traffic Volume (vehicles/hour)",
        min_value=50,
        max_value=5000,
        value=1000,
        step=50
    )
    
    speed_limit = st.select_slider(
        "Speed Limit (mph)",
        options=[25, 35, 45, 55, 65, 75],
        value=45
    )

# Prediction button
predict_button = st.button("Predict Severity", type="primary")

# Create a container for results
prediction_container = st.container()

with prediction_container:
    if predict_button:
        # Collect input data
        input_data = {
            'Day_of_Week': day,
            'Hour': hour,
            'Weather_Condition': weather,
            'Road_Type': road_type,
            'Junction_Type': junction_type,
            'Traffic_Volume': traffic_volume,
            'Speed_Limit': speed_limit
        }
        
        # Make prediction
        with st.spinner("Predicting..."):
            severity, confidence, explanation = predict_severity(st.session_state.model, input_data)
        
        # Display results
        severity_colors = {
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red'
        }
        
        # Show prediction summary
        st.markdown("### Prediction Results")
        st.markdown(f"""
        <div style="padding: 20px; background-color: {severity_colors.get(severity, 'blue')}30; border-radius: 10px;">
            <h2 style="color: {severity_colors.get(severity, 'blue')};">Predicted Severity: {severity}</h2>
            <h3>Confidence: {confidence:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display explanation
        st.markdown("### Key Factors Influencing the Prediction")
        for factor, importance in explanation:
            st.markdown(f"- **{factor}** (importance: {importance})")
        
        # Store the prediction in history
        prediction_record = input_data.copy()
        prediction_record['Prediction'] = severity
        prediction_record['Confidence'] = confidence
        prediction_record['Timestamp'] = pd.Timestamp.now()
        
        # Add to history in session state (not needed with database)
        # new_history = pd.DataFrame([prediction_record])
        # if len(st.session_state.prediction_history) > 0:
        #     st.session_state.prediction_history = pd.concat([st.session_state.prediction_history, new_history], ignore_index=True)
        # else:
        #     st.session_state.prediction_history = new_history
            
    else:
        st.info("Enter parameters and click 'Predict Severity' to get a prediction.")

# Information about the prediction model
with st.expander("About the Prediction Model"):
    st.markdown("""
    The traffic severity prediction model analyzes various factors to estimate the likely severity of a traffic incident:
    
    - **Low Severity**: Minor incidents with minimal traffic disruption, usually cleared quickly.
    - **Medium Severity**: More significant incidents requiring additional resources to clear, causing moderate traffic disruption.
    - **High Severity**: Major incidents with substantial traffic impact, potentially involving injuries and requiring extended clearance time.
    
    Factors considered include:
    
    - **Weather conditions**: Poor weather significantly increases incident severity risk.
    - **Traffic volume**: Higher traffic volumes correlate with more severe incidents.
    - **Road and junction types**: Certain road configurations are more prone to severe incidents.
    - **Time factors**: Rush hour and weekend traffic patterns affect severity outcomes.
    - **Speed limits**: Higher speed limits can lead to more severe incidents.
    """)
    
    st.markdown("Want to improve the model? You can train and integrate your own custom model! See the guide in `guides/adding_custom_model.md`.")