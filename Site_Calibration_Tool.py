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

# Function to Compute Pitch, Roll, and Heading
def compute_orientation(rtk_df, local_df):
    if len(rtk_df) < 3:
        st.error("⚠️ At least 3 reference marks are required for orientation calculation.")
        return None

    measured_points = rtk_df[["Easting", "Northing", "Height"]].values
    local_points = local_df[["X", "Y", "Z"]].values

    # Compute centroids
    centroid_measured = np.mean(measured_points, axis=0)
    centroid_local = np.mean(local_points, axis=0)

    # Center points
    measured_centered = measured_points - centroid_measured
    local_centered = local_points - centroid_local

    # Compute rotation matrix using Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(np.dot(local_centered.T, measured_centered))
    rotation_matrix = np.dot(U, Vt)

    # Ensure proper rotation (correct determinant sign)
    if np.linalg.det(rotation_matrix) < 0:
        U[:, -1] *= -1
        rotation_matrix = np.dot(U, Vt)

    # Compute Euler angles
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    roll, pitch, heading = euler_angles[0], euler_angles[1], (euler_angles[2] + 360) % 360

    return roll, pitch, heading

# Compute Orientation Button
if st.button("📊 Compute Orientation"):
    result = compute_orientation(rtk_df, local_df)
    if result:
        roll, pitch, heading = result
        st.success(f"🌀 Roll: {roll:.4f}°")
        st.success(f"🚀 Pitch: {pitch:.4f}°")
        st.success(f"🧭 Heading[GRID]: {heading:.4f}°")
