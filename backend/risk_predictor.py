from keras.models import load_model
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = load_model("model/TA_model.keras")

scaler = joblib.load("model/scaler.pkl")


def predict_risk(
    speed_limit,
    curvature,
    weather,
    road_type,
    lighting,
    traffic_density
):

    # =========================
    # CREATE FULL FEATURE VECTOR
    # =========================

    input_data = pd.DataFrame([{

        # Numerical Features
        "num_lanes": 2,
        "curvature": curvature,
        "speed_limit": speed_limit,
        "road_signs_present": 1,
        "public_road": 1,
        "holiday": 0,
        "school_season": 0,
        "num_reported_accidents": 2,

        # Road Type Encoding
        "road_type_highway": 1 if road_type == "Highway" else 0,
        "road_type_rural": 1 if road_type == "Rural" else 0,
        "road_type_urban": 1 if road_type == "Urban" else 0,

        # Lighting Encoding
        "lighting_daylight": 1 if lighting == "Day" else 0,
        "lighting_dim": 0,
        "lighting_night": 1 if lighting == "Night" else 0,

        # Weather Encoding
        "weather_clear": 1 if weather == "Clear" else 0,
        "weather_foggy": 1 if weather == "Fog" else 0,
        "weather_rainy": 1 if weather == "Rain" else 0,

        # Time of Day Encoding
        "time_of_day_afternoon": 1,
        "time_of_day_evening": 0,
        "time_of_day_morning": 0,

        # Traffic Density
        "traffic_density": traffic_density,
    }])

    # =========================
    # SCALE FEATURES
    # =========================

    scaled_features = scaler.transform(input_data)

    # Remove traffic_density before ANN prediction
    scaled_features = scaled_features[:, :-1]

    # =========================
    # MODEL PREDICTION
    # =========================

    prediction = model.predict(scaled_features)[0][0]

    return float(prediction)
