import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

st.set_page_config(page_title="Site Calibration Tool", layout="wide")  # Set page title & layout

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

st.title("📍 Site Calibration Tool (WebGUI)")

# 📌 Sidebar for Inputs
with st.sidebar:
    st.header("🔧 Input Calibration Data")
    
    # Default RTK Data
    default_rtk_data = pd.DataFrame({
        "Reference Mark": ["Ref1", "Ref2", "Ref3", "Ref4", "Ref5", "Ref6"],
        "Easting": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "Northing": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "Height": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    })

    # Default Local Data
    default_local_data = pd.DataFrame({
        "Reference Mark": ["Ref1", "Ref2", "Ref3", "Ref4", "Ref5", "Ref6"],
        "X": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "Y": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        "Z": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    })

    st.subheader("📍 Enter RTK Measurements")
    rtk_df = st.data_editor(default_rtk_data, hide_index=True, num_rows="dynamic", key="rtk_data")

    st.subheader("📍 Enter Local Site Coordinates")
    local_df = st.data_editor(default_local_data, hide_index=True, num_rows="dynamic", key="local_data")

# Function to Compute Calibration
def compute_calibration(rtk_df, local_df):
    excluded_marks = []
    valid_marks = rtk_df["Reference Mark"].tolist()  # Track valid reference marks

    while True:
        rtk_df[["Easting", "Northing", "Height"]] = rtk_df[["Easting", "Northing", "Height"]].astype(float)
        local_df[["X", "Y", "Z"]] = local_df[["X", "Y", "Z"]].astype(float)

        measured_points = rtk_df[["Easting", "Northing", "Height"]].values
        local_points = local_df[["X", "Y", "Z"]].values

        centroid_measured = np.mean(measured_points, axis=0)
        centroid_local = np.mean(local_points, axis=0)

        measured_centered = measured_points - centroid_measured
        local_centered = local_points - centroid_local

        U, S, Vt = np.linalg.svd(np.dot(local_centered.T, measured_centered))
        R_matrix = np.dot(U, Vt)

        if np.linalg.det(R_matrix) < 0:
            U[:, -1] *= -1
            R_matrix = np.dot(U, Vt)

        rotation = R.from_matrix(R_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)

        pitch = euler_angles[1]
        roll = euler_angles[0]
        heading = (euler_angles[2] + 360) % 360

        translation = centroid_local - np.dot(centroid_measured, R_matrix.T)
        transformed_points = np.dot(measured_points, R_matrix.T) + translation

        residuals = transformed_points - local_points
        horizontal_residuals = np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2)
        vertical_residuals = np.abs(residuals[:, 2])

        valid_indices = (horizontal_residuals <= 0.030) & (vertical_residuals <= 0.030)

        if np.sum(valid_indices) < 3:
            st.error("⚠️ Too few valid reference marks! At least 3 are required.")
            return None, None, None, None, None, None, None, valid_marks

        if np.all(valid_indices):
            break

        worst_index = np.argmax(horizontal_residuals + vertical_residuals)
        excluded_marks.append(valid_marks[worst_index])
        valid_marks.pop(worst_index)

        rtk_df = rtk_df.drop(index=worst_index).reset_index(drop=True)
        local_df = local_df.drop(index=worst_index).reset_index(drop=True)

        residuals = residuals[valid_indices]

    return pitch, roll, heading, residuals, R_matrix, translation, excluded_marks, valid_marks

# Compute Calibration Button
if st.button("📊 Compute Calibration"):
    pitch, roll, heading, residuals, R_matrix, translation, excluded_marks, valid_marks = compute_calibration(rtk_df, local_df)

    if residuals is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"📐 Pitch: {pitch:.4f}°")
            st.success(f"📏 Roll: {roll:.4f}°")
            st.success(f"🧭 Heading: {heading:.4f}°")

        with col2:
            if excluded_marks:
                st.warning(f"🚨 Excluded Reference Marks: {', '.join(excluded_marks)}")
            else:
                st.success("✅ No reference marks were excluded.")

        residuals_df = pd.DataFrame({
            "Reference Mark": valid_marks,
            "Horizontal Residual": np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2).round(3),
            "Vertical Residual": np.abs(residuals[:, 2]).round(3)
        })

        st.subheader("📌 Residuals per Reference Mark")
        st.data_editor(residuals_df, hide_index=True)

        csv = residuals_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Residuals as CSV", csv, "residuals.csv", "text/csv")

        with st.expander("🔍 View Raw Residuals"):
            raw_residuals_df = pd.DataFrame(residuals, columns=["Residual X", "Residual Y", "Residual Z"])
            raw_residuals_df.insert(0, "Reference Mark", valid_marks)
            st.dataframe(raw_residuals_df)

