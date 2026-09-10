import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Irrigation Need Predictor",
    page_icon="💧",
    layout="centered",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 2rem;
        }

        .title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #666;
            margin-bottom: 2rem;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 15px;
            border: 1px solid #ddd;
            margin-top: 1.5rem;
        }

        .prediction {
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            margin: 0.5rem 0 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">💧 Irrigation Need Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Enter the field and crop conditions below to estimate
        the irrigation requirement using a trained LightGBM model.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUTS
# ============================================================

st.subheader("🌱 Field Conditions")

col1, col2 = st.columns(2)

with col1:
    soil_type = st.selectbox(
        "Soil Type",
        ["Sandy", "Loamy", "Silt", "Clay"],
    )

    soil_ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.01,
    )

    soil_moisture = st.number_input(
        "Soil Moisture",
        min_value=0.0,
        max_value=100.0,
        value=32.58,
        step=0.01,
    )

    electrical_conductivity = st.number_input(
        "Electrical Conductivity",
        min_value=0.0,
        value=1.74,
        step=0.01,
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        value=26.5,
        step=0.1,
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=61.5,
        step=0.1,
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=1200.0,
        step=0.1,
    )

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=10.0,
        step=0.1,
    )


st.subheader("🌾 Crop Information")

col1, col2 = st.columns(2)

with col1:
    crop_type = st.selectbox(
        "Crop Type",
        [
            "Wheat",
            "Rice",
            "Maize",
            "Cotton",
            "Sugarcane",
        ],
    )

    growth_stage = st.selectbox(
        "Crop Growth Stage",
        [
            "Sowing",
            "Vegetative",
            "Flowering",
            "Harvest",
        ],
    )

with col2:
    season = st.selectbox(
        "Season",
        [
            "Kharif",
            "Rabi",
            "Zaid",
        ],
    )

    mulching = st.selectbox(
        "Mulching Used",
        [
            "Yes",
            "No",
        ],
    )


st.subheader("🚰 Irrigation Information")

col1, col2 = st.columns(2)

with col1:
    irrigation_type = st.selectbox(
        "Irrigation Type",
        [
            "Canal",
            "Sprinkler",
            "Rainfed",
            "Drip",
        ],
    )

    water_source = st.selectbox(
        "Water Source",
        [
            "Reservoir",
            "River",
            "Groundwater",
            "Rainwater",
        ],
    )

with col2:
    previous_irrigation = st.number_input(
        "Previous Irrigation (mm)",
        min_value=0.0,
        value=50.0,
        step=0.1,
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Irrigation Need",
    use_container_width=True,
    type="primary",
)


if predict_button:

    payload = {
        "Soil_Type": soil_type,
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Electrical_Conductivity": electrical_conductivity,
        "Temperature_C": temperature,
        "Humidity": humidity,
        "Rainfall_mm": rainfall,
        "Wind_Speed_kmh": wind_speed,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": growth_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Water_Source": water_source,
        "Mulching_Used": mulching,
        "Previous_Irrigation_mm": previous_irrigation,
    }

    with st.spinner("Analyzing field conditions..."):

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=10,
            )

            response.raise_for_status()

            result = response.json()

            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True,
            )

            st.subheader("Prediction")

            st.markdown(
                f'<div class="prediction">{prediction}</div>',
                unsafe_allow_html=True,
            )

            if prediction == "Low":
                st.success(
                    "The model predicts a low irrigation requirement."
                )
            elif prediction == "Medium":
                st.warning(
                    "The model predicts a moderate irrigation requirement."
                )
            else:
                st.error(
                    "The model predicts a high irrigation requirement."
                )

            st.write("### Prediction Probabilities")

            st.progress(
                float(probabilities["Low"]),
                text=f"Low — {probabilities['Low']:.2%}",
            )

            st.progress(
                float(probabilities["Medium"]),
                text=f"Medium — {probabilities['Medium']:.2%}",
            )

            st.progress(
                float(probabilities["High"]),
                text=f"High — {probabilities['High']:.2%}",
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure Uvicorn is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The API request timed out. Please try again."
            )

        except requests.exceptions.HTTPError as exc:
            st.error(
                f"API returned an error: {exc}"
            )

        except Exception as exc:
            st.error(
                f"Something went wrong: {exc}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by LightGBM • FastAPI • Streamlit"
)