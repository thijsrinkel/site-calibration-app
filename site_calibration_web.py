import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

st.title("Site Calibration Tool (WebGUI)")

# Initialize default RTK and Local Site Data with float-compatible values
default_rtk_data = pd.DataFrame({
    "Reference Mark": ["Ref1", "Ref2", "Ref3", "Ref4", "Ref5", "Ref6"],
    "Easting": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "Northing": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "Height": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
})

default_local_data = pd.DataFrame({
    "Reference Mark": ["Ref1", "Ref2", "Ref3", "Ref4", "Ref5", "Ref6"],
    "X": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "Y": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "Z": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
})

st.subheader("Enter RTK Measurements")
rtk_df = st.data_editor(default_rtk_data, hide_index=True, num_rows="fixed", key="rtk_data")

st.subheader("Enter Local Site Coordinates")
local_df = st.data_editor(default_local_data, hide_index=True, num_rows="fixed", key="local_data")

def compute_calibration(rtk_df, local_df):
    """ Computes Pitch, Roll, Heading, and Residuals using the Trimble Site Calibration Method. """

    # Convert DataFrame values to float
    rtk_df[["Easting", "Northing", "Height"]] = rtk_df[["Easting", "Northing", "Height"]].astype(float)
    local_df[["X", "Y", "Z"]] = local_df[["X", "Y", "Z"]].astype(float)
    
    # Extract RTK coordinates (Easting, Northing, Height)
    measured_points = rtk_df[["Easting", "Northing", "Height"]].values

    # Extract Local Site coordinates (X, Y, Z)
    local_points = local_df[["X", "Y", "Z"]].values

    # Compute residuals first to filter out bad reference marks
    transformed_points = measured_points.copy()
    residuals = transformed_points - local_points
    horizontal_residuals = np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2)  # sqrt(X^2 + Y^2)
    vertical_residuals = np.abs(residuals[:, 2])  # Z residuals

    # Identify valid reference marks (Residuals ≤ 0.03)
    valid_indices = (horizontal_residuals <= 0.03) & (vertical_residuals <= 0.03)
    
    if np.sum(valid_indices) < 3:
        st.error("Too few valid reference marks remain after filtering! At least 3 are required.")
        return None, None, None, None, None, None

    # Use only valid reference marks for calibration
    measured_points = measured_points[valid_indices]
    local_points = local_points[valid_indices]

    # Compute centroids
    centroid_measured = np.mean(measured_points, axis=0)
    centroid_local = np.mean(local_points, axis=0)

    # Center the points
    measured_centered = measured_points - centroid_measured
    local_centered = local_points - centroid_local

    # Compute optimal rotation using SVD
    U, S, Vt = np.linalg.svd(np.dot(local_centered.T, measured_centered))
    R_matrix = np.dot(U, Vt)

    # Ensure R_matrix is a proper rotation matrix
    if np.linalg.det(R_matrix) < 0:
        U[:, -1] *= -1
        R_matrix = np.dot(U, Vt)

    # Convert rotation matrix to Euler angles (XYZ convention)
    rotation = R.from_matrix(R_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=True)

    # Extract pitch, roll, and heading
    pitch = euler_angles[1]
    roll = euler_angles[0]
    heading = (euler_angles[2] + 360) % 360  # Adjust heading to 0-360°

    # Compute translation vector
    translation = centroid_local - np.dot(centroid_measured, R_matrix.T)

    # Transform measured points
    transformed_points = np.dot(measured_points, R_matrix.T) + translation

    # Compute final residuals after transformation
    residuals = transformed_points - local_points

    return pitch, roll, heading, residuals, R_matrix, translation

# Compute Calibration on Button Click
if st.button("Compute Calibration"):
    pitch, roll, heading, residuals, R_matrix, translation = compute_calibration(rtk_df, local_df)
    
    if residuals is not None:
        st.success(f"Pitch: {pitch:.4f}°")
        st.success(f"Roll: {roll:.4f}°")
        st.success(f"Heading: {heading:.4f}°")

        # Display Residuals Table
        residuals_df = pd.DataFrame({
            "Reference Mark": rtk_df["Reference Mark"],
            "Horizontal Residual": np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2).round(3),
            "Vertical Residual": np.abs(residuals[:, 2]).round(3)
        })

        st.subheader("Residuals per Reference Mark")
        st.dataframe(residuals_df)

        # Display Transformation Matrix & Translation Vector
        st.subheader("Transformation Matrix (Rotation)")
        st.write(R_matrix)

        st.subheader("Translation Vector")
        st.write(translation)

        # Prepare residuals for download
        residuals_csv = residuals_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Residuals as CSV", data=residuals_csv, file_name="residuals.csv", mime="text/csv")
