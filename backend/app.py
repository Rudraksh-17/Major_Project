# =========================
# app.py
# =========================

import os

import keras
from keras.layers import Dense

# Patch for keras compatibility
original_init = Dense.__init__


def patched_init(self, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    return original_init(self, *args, **kwargs)


Dense.__init__ = patched_init

import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime

# Custom modules
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
.media-box {
    border-radius: 15px;
    background-color: #111827;
    padding: 15px;
    margin-bottom: 10px;
}
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("""
<div style="
padding:20px;
border-radius:20px;
background:linear-gradient(90deg,#111827,#1f2937);
text-align:center;
">

<h1 style="color:white;">
AADMTA-AI Assisted Driver Monitoring and Traffic Analytics
</h1>

<p style="color:#9ca3af;font-size:18px;">
AI Tribrid Transportation Safety Platform
</p>

</div>
""",
unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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

# =========================
# EYE IMAGE UPLOAD
# =========================

uploaded_eye = st.sidebar.file_uploader(
    "Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# VIDEO UPLOAD
# =========================

uploaded_video = st.sidebar.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

predict_button = st.sidebar.button("Predict Risk")

drowsiness_score = 0.0
risk_level = "Not Predicted"
risk_class = "risk-low"
overall_risk = 0.0

# =========================
# MAIN LAYOUT
# =========================
k1,k2,k3,k4 = st.columns(4)

with k1:
    st.metric(
        "CNN Model",
        "ACTIVE"
    )

with k2:
    st.metric(
        "YOLO Model",
        "ACTIVE"
    )

with k3:
    st.metric(
        "ANN Model",
        "ACTIVE"
    )

with k4:
    st.metric(
        "System",
        "ONLINE"
    )
media_col1, media_col2 = st.columns([1,1], gap="large")

# =========================
# LEFT PANEL
# =========================

# =========================
# MEDIA SECTION
# =========================

with media_col1:

    st.markdown("""
    <div class="media-box">
    <h3>📹 Traffic Monitoring</h3>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_video is not None:

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        st.video("temp_video.mp4")

    else:

        st.info(
            "Upload a traffic video for vehicle detection."
        )

with media_col2:

    st.markdown("""
    <div class="media-box">
    <h3>👁 Driver Monitoring</h3>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_eye is not None:

        uploaded_eye.seek(0)

        image = Image.open(uploaded_eye)

        st.image(
            image,
            use_container_width=True
        )

    else:

        st.info(
            "Upload a driver eye image."
        )

# =========================
# TRAFFIC PROCESSING
# =========================

if uploaded_video is not None:

    vehicle_count, traffic_density = detect_traffic(
        "temp_video.mp4"
    )

else:

    vehicle_count = 0
    traffic_density = 0.1

if traffic_density < 0.4:
    traffic_label = "Low"

elif traffic_density < 0.7:
    traffic_label = "Medium"

else:
    traffic_label = "High"

# =========================
# ANALYTICS SECTION
# =========================

analytics_col1, analytics_col2 = st.columns(2)

with analytics_col1:

    st.subheader("📊 Traffic Intelligence")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric(
            "Vehicle Count",
            vehicle_count
        )

    with stat2:
        st.metric(
            "Traffic Density",
            traffic_label
        )

    with stat3:
        st.metric(
            "System Status",
            "Active"
        )

with analytics_col2:

    st.markdown("""
    <div style="
    background:#111827;
    padding:15px;
    border-radius:15px;
    margin-bottom:15px;
    ">
    <h3 style="color:white;">
    🧠 Transportation Risk Engine
    </h3>
    </div>
    """, unsafe_allow_html=True)

    if predict_button:

        prediction = predict_risk(
            speed_limit=speed_limit,
            curvature=curvature,
            weather=weather,
            road_type=road_type,
            lighting=lighting,
            traffic_density=traffic_density
        )

        if uploaded_eye is not None:
            uploaded_eye.seek(0)

            image_bytes = uploaded_eye.getvalue()
            with open("temp_eye.jpg", "wb") as f:
                f.write(image_bytes)
                drowsiness_score = detect_drowsiness("temp_eye.jpg")
    

        overall_risk = (
            prediction * 0.3 +
            traffic_density * 0.4 +
            drowsiness_score * 0.3
        )

        if overall_risk < 0.3:

            risk_level = "LOW"
            risk_class = "risk-low"

        elif overall_risk < 0.5:

            risk_level = "MEDIUM"
            risk_class = "risk-medium"

        else:

            risk_level = "HIGH"
            risk_class = "risk-high"

        st.markdown(f"""
        <div class="metric-card">
            <h2>Predicted Risk</h2>
            <h1 class="{risk_class}">
                {risk_level}
            </h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Hybrid Risk Score")

        st.progress(
            float(min(overall_risk, 1.0))
        )

        st.metric(
            "Overall Risk Probability",
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

st.dataframe(
    history_data,
    use_container_width=True
)
