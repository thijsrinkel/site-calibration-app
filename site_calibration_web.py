import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

st.title("Site Calibration Tool (WebGUI)")

# Upload RTK and Local Coordinate CSV files
rtk_file = st.file_uploader("Upload RTK Measurement File (CSV)", type=["csv"])
local_file = st.file_uploader("Upload Local Coordinates File (CSV)", type=["csv"])

def compute_calibration(rtk_df, local_df):
    """ Computes Pitch, Roll, and Heading using the Trimble Site Calibration Method. """
    
    # Ensure valid data
    if rtk_df.shape[0] != 6 or local_df.shape[0] != 6:
        st.error("Each file must contain exactly 6 reference marks.")
        return None

    # Extract RTK coordinates (Easting, Northing, Height)
    measured_points = rtk_df[["Easting", "Northing", "Height"]].values

    # Extract Local Site coordinates (X, Y, Z)
    local_points = local_df[["X", "Y", "Z"]].values

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

    return pitch, roll, heading

# Process files when both are uploaded
if rtk_file and local_file:
    rtk_df = pd.read_csv(rtk_file)
    local_df = pd.read_csv(local_file)

    if st.button("Compute Calibration"):
        results = compute_calibration(rtk_df, local_df)
        
        if results:
            pitch, roll, heading = results
            st.success(f"Pitch: {pitch:.4f}°")
            st.success(f"Roll: {roll:.4f}°")
            st.success(f"Heading: {heading:.4f}°")

            # Prepare results for download
            results_df = pd.DataFrame({"Pitch (°)": [pitch], "Roll (°)": [roll], "Heading (°)[GRID]": [heading]})
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Results as CSV", data=csv_data, file_name="calibration_results.csv", mime="text/csv")
