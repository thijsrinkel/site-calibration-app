import streamlit as st
import pandas as pd
import numpy as np
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
st.title("📍 Site Calibration Tool")

# 📖 User Guide
with st.expander("ℹ️ **How to Use This Tool**", expanded=False):
    st.markdown("""
    Welcome to the **Site Calibration Tool**! Follow these steps:

    1️⃣ **Enter Your Data:**
    - Input the Topo measurements (Easting, Northing, Height).
    - Enter the Local Caisson Coordinates (X, Y, Z).

    2️⃣ **Click 'Compute Calibration':**
    - The tool will calculate **pitch, roll, heading, and residuals**.
    - If **reference marks exceed the threshold**, they are **excluded**.

    3️⃣ **Review the Results:**
    - The **residuals per reference mark** will be displayed.
    - Excluded marks will be shown as a warning.

    4️⃣ **Download the Results (Optional):**
    - Click **"⬇️ Download Residuals as CSV"** to save.

    ⚠️ **Minimum 3 reference marks required!**
    """)

# 📌 Sidebar for Inputs
with st.sidebar:
    st.image("TM_Edison_logo.jpg", width=150)
    st.header("🔧 Input Calibration Data")

    default_rtk_data = pd.DataFrame({
        "Reference Mark": [f"Ref{i+1}" for i in range(6)],
        "Easting": [0.000] * 6,
        "Northing": [0.000] * 6,
        "Height": [0.000] * 6
    })

    default_local_data = pd.DataFrame({
        "Reference Mark": [f"Ref{i+1}" for i in range(6)],
        "X": [0.000] * 6,
        "Y": [0.000] * 6,
        "Z": [0.000] * 6
    })

    st.subheader("📍 Enter Topo Measurements")
    rtk_df = st.data_editor(
        default_rtk_data, hide_index=True, num_rows="dynamic", key="rtk_data",
        column_config={
            "Easting": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Northing": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Height": st.column_config.NumberColumn(format="%.3f", step=0.001),
        }
    )

    st.subheader("📍 Enter Local Caisson Coordinates")
    local_df = st.data_editor(
        default_local_data, hide_index=True, num_rows="dynamic", key="local_data",
        column_config={
            "X": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Y": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Z": st.column_config.NumberColumn(format="%.3f", step=0.001),
        }
    )

# ✅ **Fix: Improve Robustness of Compute Function**
def compute_calibration(rtk_df, local_df):
    if len(rtk_df) < 3:
        st.error("⚠️ You need at least 3 reference marks to compute calibration.")
        return None, None, None, None, None, None, None, None

    excluded_marks = []
    valid_marks = rtk_df["Reference Mark"].tolist()

    while len(valid_marks) >= 3:
        rtk_df[["Easting", "Northing", "Height"]] = rtk_df[["Easting", "Northing", "Height"]].astype(float)
        local_df[["X", "Y", "Z"]] = local_df[["X", "Y", "Z"]].astype(float)

        measured_points = rtk_df[["Easting", "Northing", "Height"]].values
        local_points = local_df[["X", "Y", "Z"]].values

        if np.all(measured_points == measured_points[0]) or np.all(local_points == local_points[0]):
            st.error("❌ Error: All input points are identical. Ensure at least 3 distinct reference marks.")
            return None, None, None, None, None, None, None, None

        centroid_measured = np.mean(measured_points, axis=0)
        centroid_local = np.mean(local_points, axis=0)

        measured_centered = measured_points - centroid_measured
        local_centered = local_points - centroid_local

        try:
            U, S, Vt = np.linalg.svd(np.dot(local_centered.T, measured_centered))
        except np.linalg.LinAlgError:
            st.error("❌ SVD Computation Error: Input points may be collinear. Adjust input data.")
            return None, None, None, None, None, None, None, None

        R_matrix = np.dot(U, Vt)

        if np.linalg.det(R_matrix) < 0:
            U[:, -1] *= -1
            R_matrix = np.dot(U, Vt)

        rotation = R.from_matrix(R_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        pitch, roll, heading = euler_angles[1], euler_angles[0], (euler_angles[2] + 360) % 360

        translation = centroid_local - np.dot(centroid_measured, R_matrix.T)
        transformed_points = np.dot(measured_points, R_matrix.T) + translation

        residuals = transformed_points - local_points
        horizontal_residuals = np.linalg.norm(residuals[:, :2], axis=1)
        vertical_residuals = np.abs(residuals[:, 2])

        valid_indices = (horizontal_residuals <= 0.030) & (vertical_residuals <= 0.030)

        if np.sum(valid_indices) < 3:
            st.error("⚠️ Too few valid reference marks! At least 3 are required.")
            return None, None, None, None, None, None, excluded_marks, valid_marks

        if np.all(valid_indices):
            break

        worst_index = np.argmax(horizontal_residuals + vertical_residuals)
        excluded_marks.append(valid_marks.pop(worst_index))

        rtk_df = rtk_df.drop(index=worst_index).reset_index(drop=True)
        local_df = local_df.drop(index=worst_index).reset_index(drop=True)

        residuals = residuals[valid_indices]

    return pitch, roll, heading, residuals, R_matrix, translation, excluded_marks, valid_marks

if st.button("📊 Compute Calibration"):
    pitch, roll, heading, residuals, R_matrix, translation, excluded_marks, valid_marks = compute_calibration(rtk_df, local_df)

    if residuals is not None:
        st.success(f"🚀 Pitch: {pitch:.4f}° | 🌀 Roll: {roll:.4f}° | 🧭 Heading: {heading:.4f}°")
