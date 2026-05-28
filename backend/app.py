import keras
from keras.layers import Dense



# Capture the original initialization method
original_init = Dense.__init__

# Define a patched initialization that filters out the breaking argument
def patched_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)  # Safely strip the key if present
    return original_init(self, *args, **kwargs)

# Overwrite the default Keras Dense layer initialization logic
Dense.__init__ = patched_init

import streamlit as st
import pandas as pd
from datetime import datetime

# Import custom modules
from traffic_detection import detect_traffic
from risk_predictor import predict_risk
from Drowsiness_detector import detect_drowsiness

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Transportation Risk System",
    page_icon="🚗",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.3);
}

.risk-high {
    color: red;
    font-size: 32px;
    font-weight: bold;
}

.risk-medium {
    color: orange;
    font-size: 32px;
    font-weight: bold;
}

.risk-low {
    color: lightgreen;
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.title("🚗 AI Transportation Risk Analysis System")

st.markdown("---")

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("⚙️ Input Parameters")

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Cloudy", "Rain", "Fog"]
)

road_type = st.sidebar.selectbox(
    "Road Type",
    ["Urban", "Highway", "Rural"]
)

lighting = st.sidebar.selectbox(
    "Lighting",
    ["Day", "Night"]
)

speed_limit = st.sidebar.slider(
    "Speed Limit",
    20,
    120,
    60
)

curvature = st.sidebar.slider(
    "Road Curvature",
    0.0,
    1.0,
    0.3
)

uploaded_eye = st.sidebar.file_uploader(
    "Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)
uploaded_video = st.sidebar.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)
drowsiness_score = 0.0
predict_button = st.sidebar.button("Predict Risk")

# =========================
# MAIN LAYOUT
# =========================

col1, col2 = st.columns([2, 1])

# =========================
# LEFT PANEL
# =========================

with col1:

    st.subheader("📹 Traffic Monitoring Feed")

if uploaded_video is not None:

    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.video("temp_video.mp4")

else:

    st.image(
        "https://images.unsplash.com/photo-1502877338535-766e1452684a",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📊 Traffic Statistics")

    # =========================
    # YOLO DETECTION
    # =========================

    if uploaded_video is not None:

    vehicle_count, traffic_density = detect_traffic("temp_video.mp4")

    else:

    vehicle_count = 0
    traffic_density = 0.1
    # Density label
    if traffic_density < 0.4:
        traffic_label = "Low"

    elif traffic_density < 0.7:
        traffic_label = "Medium"

    else:
        traffic_label = "High"

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric("Vehicle Count", vehicle_count)

    with stat2:
        st.metric("Traffic Density", traffic_label)

    with stat3:
        st.metric("System Status", "Active")

# =========================
# RIGHT PANEL
# =========================

with col2:

    st.subheader("🧠 AI Risk Analysis")

    if predict_button:

        # =========================
        # ANN PREDICTION
        # =========================

        prediction = predict_risk(
            speed_limit=speed_limit,
            curvature=curvature,
            weather=weather,
            road_type=road_type,
            lighting=lighting,
            traffic_density=traffic_density
        )

        if uploaded_eye is not None:

            with open("temp_eye.jpg", "wb") as f:
                f.write(uploaded_eye.read())

            # =========================
            # Run drowsiness detection after saving the uploaded image
            drowsiness_score = detect_drowsiness("temp_eye.jpg")

            # =========================
            # HYBRID RISK CALCULATION
        # =========================

        overall_risk = (
            prediction * 0.4 +
            traffic_density * 0.3 +
            drowsiness_score * 0.3
        )

        # =========================
        # RISK CLASSIFICATION
        # =========================

        if overall_risk < 0.3:
            risk_level = "LOW"
            risk_class = "risk-low"

        elif overall_risk < 0.5:
            risk_level = "MEDIUM"
            risk_class = "risk-medium"

        else:
            risk_level = "HIGH"
            risk_class = "risk-high"

        # =========================
        # DISPLAY RESULTS
        # =========================

        st.markdown(f"""
        <div class="metric-card">
            <h2>Predicted Risk</h2>
            <h1 class="{risk_class}">
                {risk_level}
            </h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Risk Score")

        st.progress(float(min(overall_risk, 1.0)))

        st.metric(
            " Overall Risk Probability",
            f"{overall_risk:.2f}"
        )

        st.metric(
            "Weather",
            weather
        )

        st.metric(
            "Road Type",
            road_type
        )

        st.metric(
            "Lighting",
            lighting
        )

# =========================
# PREDICTION HISTORY
# =========================

st.markdown("---")

st.subheader("🗂️ Prediction History")

history_data = pd.DataFrame({
    "Time": [
        datetime.now().strftime("%H:%M:%S")
    ],
    "Weather": [weather],
    "Road Type": [road_type],
    "Traffic Density": [traffic_label],
    "Risk": [
        risk_level if predict_button else "Not Predicted"
    ]
})

st.dataframe(history_data, use_container_width=True)


