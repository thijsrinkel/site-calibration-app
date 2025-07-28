import numpy as np
import pandas as pd
import streamlit as st

from kabsch import best_fit_transform, get_rotation_error, rotation_matrix_to_rpy


st.set_page_config(page_title="Site Calibration Tool", layout="wide")

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

st.image("img/TM_Edison_logo.jpg", width=150)
st.title("📍 Site Calibration Tool 2.0")

with st.sidebar:
    st.header("🔧 Input Calibration Data")

    default_rtk_data = pd.DataFrame(
        {
            "Reference Mark": ["Ref 1", "Ref 2", "Ref 3", "Ref 4", "Ref 5", "Ref 6"],
            "Easting": [np.nan] * 6,
            "Northing": [np.nan] * 6,
            "Height": [np.nan] * 6,
        }
    )

    default_local_data = pd.DataFrame(
        {
            "Reference Mark": ["Ref 1", "Ref 2", "Ref 3", "Ref 4", "Ref 5", "Ref 6"],
            "X": [np.nan] * 6,
            "Y": [np.nan] * 6,
            "Z": [np.nan] * 6,
        }
    )

    st.subheader("📍 Enter Topo Measurements")
    rtk_df = st.data_editor(
        default_rtk_data,  # Keep "Reference Mark" editable
        hide_index=True,  # Hides the default Streamlit index
        key="rtk_data",
        column_config={
            "Reference Mark": st.column_config.TextColumn(),  # Editable reference marks
            "Easting": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Northing": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Height": st.column_config.NumberColumn(format="%.3f", step=0.001),
        },
    )

    st.subheader("📍 Enter Local Caisson Coordinates")
    local_df = st.data_editor(
        default_local_data,  # Keep "Reference Mark" editable
        hide_index=True,  # Hides the default Streamlit index
        key="local_data",
        column_config={
            "Reference Mark": st.column_config.TextColumn(),  # Editable reference marks
            "X": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Y": st.column_config.NumberColumn(format="%.3f", step=0.001),
            "Z": st.column_config.NumberColumn(format="%.3f", step=0.001),
        },
    )

with st.expander("ℹ️ **How to Use This Tool**", expanded=False):
    st.markdown(
        """
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

    """
    )

if st.button("📊 Compute Calibration"):

    rtk_refs_to_use = np.array(rtk_df.iloc[:, 1:].notna().all(axis=1))
    local_refs_to_use = np.array(local_df.iloc[:, 1:].notna().all(axis=1))
    refs_to_use = rtk_refs_to_use & local_refs_to_use

    if sum(refs_to_use) < 3:
        st.warning("Need at least 3 points!")
    else:
        rtk_refs = rtk_df[refs_to_use]
        local_refs = local_df[refs_to_use]
        ref_marks = rtk_refs["Reference Mark"].values
        st.write("Using " + ", ".join(ref_marks[:-1]) + " and " + ref_marks[-1])

        measured_points = rtk_refs[["Easting", "Northing", "Height"]].values
        local_points = local_refs[["X", "Y", "Z"]].values

        R, t = best_fit_transform(local_points, measured_points)

        error = get_rotation_error(local_points, measured_points, R, t)

        roll, pitch, yaw = rotation_matrix_to_rpy(R)
        heading = (360 - yaw) % 360

        st.success(f"🚀 Pitch: {pitch:.3f}°")
        st.success(f"🌀 Roll: {roll:.3f}°")
        st.success(f"🧭 Heading[GRID]: {heading:.3f}°")

        st.write("Distance of rotated local points to topo points: ")
        df_error = pd.DataFrame(
            np.column_stack([ref_marks, error]), columns=["Reference Mark", "Distance"]
        )
        st.dataframe(df_error, hide_index=True)
