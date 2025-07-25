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
    - Enter the Local Caisson (QINSY) Coordinates (X, Y, Z).

    2️⃣ **Click 'Compute Calibration':**
    - The tool will calculate **pitch, roll, heading, and residuals**.
    - If **Residual of the reference marks exceed the threshold of 0.03**, they are **excluded**.

    3️⃣ **Review the Results:**
    - The **residuals per reference mark** will be displayed.
    - Excluded marks will be shown as a warning.

    4️⃣ **Download the Results (Optional):**
    - Click **"⬇️ Download Residuals as CSV"** to save.

        🌍 **Conventions**
    - **Roll = Positive** → **Heeling to Starboard**.
    - **Pitch = Positive** → **Bow Up**.
    - **Heading = Grid north**.
    - **X = Positive** → **Starboard**.
    - **Y = Positive** → **Bow**.
    - **Z = Positive** → **Up**.
    ---
    **Tips:**
    - Ensure that at least **3 valid reference marks** remain after filtering.
    - If too many reference marks are removed, try adjusting your input data.
    - Ensure that your values have at least 2 decimals, preferably 3.
    
    """)

# 📌 Sidebar for Inputs
with st.sidebar:
    st.image("TR-offshore-survey-logo-1-200x75.png", width=200)
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
    rtk_df = st.data_editor(
        default_rtk_data,  # Keep "Reference Mark" editable
        hide_index=True,   # Hides the default Streamlit index
        key="rtk_data",
        column_config={
            "Reference Mark": st.column_config.TextColumn(),  # Editable reference marks
            "Easting": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Northing": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Height": st.column_config.NumberColumn(format="%.3f", step=0.001),
        }
    )

    st.subheader("📍 Enter Local Caisson Coordinates")
    local_df = st.data_editor(
        default_local_data,  # Keep "Reference Mark" editable
        hide_index=True,   # Hides the default Streamlit index
        key="local_data",
        column_config={
            "Reference Mark": st.column_config.TextColumn(),  # Editable reference marks
            "X": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Y": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Z": st.column_config.NumberColumn(format="%.3f", step=0.001),
        }
    )


# Function to Compute Calibration
def compute_calibration(rtk_df, local_df):
    excluded_marks = []
    valid_marks = rtk_df["Reference Mark"].tolist()  

    while len(valid_marks) >= 3:
        rtk_df[["Easting", "Northing", "Height"]] = rtk_df[["Easting", "Northing", "Height"]].astype(float)
        local_df[["X", "Y", "Z"]] = local_df[["X", "Y", "Z"]].astype(float)

        measured_points = rtk_df[["Easting", "Northing", "Height"]].values
        local_points = local_df[["X", "Y", "Z"]].values

        # Compute centroids
        centroid_measured = np.mean(measured_points, axis=0)
        centroid_local = np.mean(local_points, axis=0)

        # Center points
        measured_centered = measured_points - centroid_measured
        local_centered = local_points - centroid_local

        # Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(np.dot(local_centered.T, measured_centered))
        R_matrix = np.dot(U, Vt)

        # Ensure proper rotation (correct determinant sign)
        if np.linalg.det(R_matrix) < 0:
            U[:, -1] *= -1
            R_matrix = np.dot(U, Vt)

        # Step 1: Full rotation from measured to local
        rotation = R.from_matrix(R_matrix)

        # Step 2: Extract heading (rotation around Z in global frame)
        heading_rad = rotation.as_euler('xyz', degrees=False)[2]
        heading = np.degrees(heading_rad) % 360

        # Step 3: Remove heading (rotate back to vessel frame)
        # In quaternion form: R_heading_inv * R_full
        deheaded_rotation = R.from_euler('z', -heading_rad) * rotation

        # Step 4: Extract pitch and roll in vessel-aligned frame
        # Now in 'yx' order: pitch about Y (bow), roll about X (starboard)
        pitch, roll = deheaded_rotation.as_euler('yx', degrees=True)

        # Compute translation
        translation = centroid_local - np.dot(centroid_measured, R_matrix.T)
        transformed_points = np.dot(measured_points, R_matrix.T) + translation

        # Compute residuals
        residuals = transformed_points - local_points
        horizontal_residuals = np.linalg.norm(residuals[:, :2], axis=1)
        vertical_residuals = np.abs(residuals[:, 2])

        # Check which marks exceed threshold
        valid_indices = (horizontal_residuals <= 0.030) & (vertical_residuals <= 0.030)

        if np.sum(valid_indices) < 3:
            st.error("⚠️ Too few valid reference marks! At least 3 are required.")
            return None, None, None, None, None, None, excluded_marks, valid_marks

        if np.all(valid_indices):
            break  # Exit loop if all marks are valid

        # Identify worst mark to exclude
        worst_index = np.argmax(horizontal_residuals + vertical_residuals)
        excluded_marks.append(valid_marks.pop(worst_index))

        # Drop the worst index from dataframes
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
            st.success(f"🚀 Pitch: {pitch:.4f}°")
            st.success(f"🌀 Roll: {roll:.4f}°")
            st.success(f"🧭 Heading[GRID]: {heading:.4f}°")

        with col2:
            if excluded_marks:
                st.warning(f"🚨 Excluded Reference Marks: {', '.join(excluded_marks)}")
            else:
                st.success("✅ No reference marks were excluded.")

        residuals_df = pd.DataFrame({
            "Reference Mark": valid_marks,
            "Horizontal Residual": np.round(np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2), 3),
            "Vertical Residual": np.round(np.abs(residuals[:, 2]), 3)
        })

        st.subheader("📌 Residuals per Reference Mark")
        st.data_editor(residuals_df, hide_index=True)

        csv = residuals_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Residuals as CSV", csv, "residuals.csv", "text/csv")

        with st.expander("🔍 View Raw Residuals"):
            raw_residuals_df = pd.DataFrame(residuals, columns=["Residual X", "Residual Y", "Residual Z"])
            raw_residuals_df.insert(0, "Reference Mark", valid_marks)
            st.dataframe(raw_residuals_df)
