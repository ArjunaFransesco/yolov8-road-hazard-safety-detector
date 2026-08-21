import streamlit as st
import pandas as pd
import numpy as np
import joblib, os

st.set_page_config(page_title="Road Hazard Detector", page_icon="🚗", layout="wide")
st.title("🚗 Real-Time Road Hazard & Pothole Diagnostic Engine")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **Portfolio**: [GitHub](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Computer Vision & Telemetry Inputs")
    area = st.slider("Bounding Box Area (px)", 50, 8000, 3200)
    aspect = st.slider("Aspect Ratio (W/H)", 0.2, 5.0, 1.4)
    contrast = st.slider("Intensity Contrast (0-100)", 5.0, 95.0, 62.0)
    edge = st.slider("Edge Density Metric", 0.05, 0.95, 0.45)
    depth = st.slider("Estimated Depth (cm)", 0.0, 25.0, 8.5)

with col2:
    st.subheader("🔍 Diagnostic Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/road_hazard_classifier.joblib")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        input_data = pd.DataFrame([{
            "bounding_box_area_px": area,
            "aspect_ratio": aspect,
            "intensity_contrast": contrast,
            "edge_density": edge,
            "depth_est_cm": depth
        }])
        pred = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        
        st.success(f"### Detected Hazard: **{pred}**")
        prob_df = pd.DataFrame({"Hazard": model.classes_, "Confidence": probs})
        st.bar_chart(prob_df.set_index("Hazard"))
        
        urgency = np.clip((depth / 20.0) * 0.7 + (area / 7000.0) * 0.3, 0.0, 1.0)
        st.metric("Repair Urgency Score", f"{urgency:.1%}")
    else:
        st.warning("Model artifact not loaded.")
