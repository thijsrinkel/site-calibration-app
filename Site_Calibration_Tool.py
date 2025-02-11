import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

st.set_page_config(page_title="Site Calibration Tool", layout="wide")

# 🎨 Custom Styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px;
    }
    .stDataFrame { font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

st.image("TM_Edison_logo.jpg", width=150)
st.title("📍 Trimble-Based Site Calibration Tool")

# 📌 Sidebar for Inputs
with st.sidebar:
    st.image("TM_Edison_logo.jpg", width=150)
    st.header("🔧 Input Calibration Data")

    default_rtk_data = pd.DataFrame({
        "Reference Mark": ["Ref 1", "Ref 2", "Ref 3", "Ref 4", "Ref 5", "Ref 6"],
        "Easting": [0.000] * 6,
        "Northing": [0.000] * 6,
        "Height": [0.000] * 6
    })

    default_local_data = pd.DataFrame({
        "Reference Mark": ["Ref 1", "Ref 2", "Ref 3", "Ref 4", "Ref 5", "Ref 6"],
        "X": [0.000] * 6,
        "Y": [0.000] * 6,
        "Z": [0.000] * 6
    })

    st.subheader("📍 Enter Topo Measurements")
    rtk_df = st.data_editor(default_rtk_data, hide_index=True, key="rtk_data")
    st.subheader("📍 Enter Local Caisson Coordinates")
    local_df = st.data_editor(default_local_data, hide_index=True, key="local_data")

# Function to Compute Trimble-Based Calibration
def compute_trimble_calibration(rtk_df, local_df):
    if len(rtk_df) < 4:
        st.error("⚠️ At least 4 reference marks are required for accurate calibration.")
        return None

    measured_points = rtk_df[["Easting", "Northing", "Height"]].values
    local_points = local_df[["X", "Y", "Z"]].values

    def residuals(params, measured, local):
        scale = params[0]
        rotation_vector = params[1:4]
        translation_vector = params[4:]
        rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
        transformed_points = scale * (measured @ rotation_matrix.T) + translation_vector
        return (transformed_points - local).flatten()

    # Use Trimble-provided scale factor
    fixed_scale_factor = np.sum(np.linalg.norm(local_points[1:] - local_points[:-1], axis=1)) / np.sum(np.linalg.norm(measured_points[1:] - measured_points[:-1], axis=1))
    initial_guess = np.array([fixed_scale_factor, 0.0, 0.0, 0.0, *(np.mean(local_points, axis=0) - np.mean(measured_points, axis=0))])
    result = least_squares(residuals, initial_guess, args=(measured_points, local_points))
    
    optimized_rotation = result.x[1:4]
    optimized_translation = result.x[4:]
    rotation_matrix = R.from_rotvec(optimized_rotation).as_matrix()

    # Ensure correct handedness of rotation
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, -1] *= -1  # Correct determinant issue

    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    roll, pitch, heading = euler_angles[0], euler_angles[1], (euler_angles[2] + 360) % 360

    # Convert heading to degrees, minutes, seconds format (DMS)
    heading_deg = int(heading)
    heading_min = int((heading - heading_deg) * 60)
    heading_sec = (heading - heading_deg - heading_min / 60) * 3600
    heading_dms = f"{heading_deg}° {heading_min}' {heading_sec:.2f}\""

    # Compute refined translation after applying rotation and scaling
    transformed_points = fixed_scale_factor * (measured_points @ rotation_matrix.T) + optimized_translation
    refined_translation = np.mean(local_points - transformed_points, axis=0)

    return fixed_scale_factor, roll, pitch, heading_dms, refined_translation

# Compute Calibration Button
if st.button("📊 Compute Calibration"):
    result = compute_trimble_calibration(rtk_df, local_df)
    if result:
        scale_factor, roll, pitch, heading, translation = result
        st.success(f"📏 Computed Scale Factor: {scale_factor:.8f}")
        st.success(f"🌀 Roll: {roll:.4f}°")
        st.success(f"🚀 Pitch: {pitch:.4f}°")
        st.success(f"🧭 Heading[GRID]: {heading}")
        st.success(f"📍 Translation Adjustment: X={translation[0]:.4f}, Y={translation[1]:.4f}, Z={translation[2]:.4f}")
