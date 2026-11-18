
# ============================================================
# AIR QUALITY MANAGEMENT AND PREDICTION SYSTEM
# MCA FINAL YEAR PROJECT
# AGR STUDYORBIT
# STREAMLIT DEPLOYMENT VERSION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from lime.lime_tabular import LimeTabularExplainer
from datetime import date

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Air Quality Management System",
    page_icon="🌍",
    layout="wide"
)

# ------------------------------------------------------------
# 2. CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #f1f6fc;
}

h1, h2, h3 {
    color: #12345a;
}

div.stButton > button {
    background-color: #0077b6;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 3. LOAD TRAINED MODEL
# ------------------------------------------------------------

MODEL_PATH = "air_quality_model.pkl"

@st.cache_resource
def load_model():

    pipeline = joblib.load(MODEL_PATH)

    model = pipeline["model"]
    preprocessor = pipeline["preprocessor"]
    feature_names = pipeline["feature_names"]
    lime_training_data = pipeline["lime_training_data"]

    return (
        model,
        preprocessor,
        feature_names,
        lime_training_data
    )

try:
    model, preprocessor, feature_names, lime_training_data = load_model()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ------------------------------------------------------------
# 4. APPLICATION HEADER
# ------------------------------------------------------------

st.markdown("""
<div style="
    background: linear-gradient(90deg,#12345a,#2878bd);
    padding: 25px;
    border-radius: 14px;
    text-align: center;
">
<h1 style="color:orange;">
🌍 AIR QUALITY MANAGEMENT SYSTEM
</h1>

<h3 style="color:white;">
AI-Based Air Quality Management and Prediction
System Using Machine Learning
</h3>
</div>
""", unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Predictor",
        "SHAP & LIME",
        "Input Guide"
    ],
    key="page"

)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Project Information

**Domain:** Big Data + Machine Learning

**Model:** Decision Tree Classifier

**Interface:** Streamlit

**Project:** AGR STUDYORBIT
""")

# ------------------------------------------------------------
# 6. HOME PAGE
# ------------------------------------------------------------

if page == "Home":

    st.title("🏠 Home")

    st.markdown("""
    ## Welcome to the Air Quality Management and Prediction System

    This project uses Machine Learning to predict
    air quality categories based on air pollution
    measurements.

    ### 🎯 Project Objectives

    - Analyse air pollution measurements.
    - Predict air quality categories.
    - Display prediction results.
    - Explain model predictions using SHAP and LIME.
    - Support better understanding of air quality.

    ### ⚙️ Project Workflow

    1. Dataset collection
    2. Data cleaning and preprocessing
    3. Exploratory data analysis
    4. Machine Learning model training
    5. Model evaluation
    6. Air quality prediction
    7. Explainable AI using SHAP and LIME
    """)

    if st.button("🚀 Start Air Quality Prediction"):
        st.session_state["page"] = "Predictor"
        st.rerun()

# ------------------------------------------------------------
# 7. PREDICTOR PAGE
# ------------------------------------------------------------

elif page == "Predictor":

    st.title("🔍 Air Quality Predictor")

    st.write(
        "Enter the air quality measurements below "
        "and click Predict."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            city = st.text_input(
                "City",
                value="Delhi"
            )

            selected_date = st.date_input(
                "Date",
                value=date.today()
            )

            pm25_mean = st.number_input(
                "PM2.5 Mean",
                value=80.0
            )

            pm25_max = st.number_input(
                "PM2.5 Maximum",
                value=120.0
            )

            pm10_mean = st.number_input(
                "PM10 Mean",
                value=150.0
            )

            pm10_max = st.number_input(
                "PM10 Maximum",
                value=200.0
            )

            no2_mean = st.number_input(
                "NO2 Mean",
                value=40.0
            )

        with col2:

            so2_mean = st.number_input(
                "SO2 Mean",
                value=15.0
            )

            o3_max = st.number_input(
                "O3 Maximum",
                value=50.0
            )

            co_max = st.number_input(
                "CO Maximum",
                value=1.2
            )

            dust_mean = st.number_input(
                "Dust Mean",
                value=80.0
            )

            aod_mean = st.number_input(
                "AOD Mean",
                value=0.5
            )

            hours_available = st.slider(
                "Hours Available",
                min_value=0,
                max_value=24,
                value=24
            )

        submitted = st.form_submit_button(
            "🔮 Predict Air Quality"
        )

    # --------------------------------------------------------
    # 8. PREDICTION PROCESS
    # --------------------------------------------------------

    if submitted:

        if not city.strip():

            st.error("Please enter a city name.")

        else:

            try:

                input_data = pd.DataFrame([{

                    "city": city.strip(),

                    "pm2_5_mean": pm25_mean,
                    "pm2_5_max": pm25_max,

                    "pm10_mean": pm10_mean,
                    "pm10_max": pm10_max,

                    "no2_mean": no2_mean,
                    "so2_mean": so2_mean,

                    "o3_8h_max": o3_max,
                    "co_8h_max": co_max,

                    "dust_mean": dust_mean,
                    "aod_mean": aod_mean,

                    "hours_available": hours_available,

                    "year": selected_date.year,
                    "month": selected_date.month,
                    "day": selected_date.day,
                    "day_of_week": selected_date.weekday()

                }])

                # Apply saved preprocessing

                processed = preprocessor.transform(
                    input_data
                )

                # Predict category

                prediction = model.predict(
                    processed
                )[0]

                # Calculate probabilities

                probabilities = model.predict_proba(
                    processed
                )[0]

                probability_df = pd.DataFrame({

                    "AQI Category": model.classes_,

                    "Probability (%)": np.round(
                        probabilities * 100, 2
                    )

                })

                st.session_state["input_data"] = input_data

                st.session_state["prediction"] = prediction

                st.session_state["probability_df"] = probability_df

                st.success("Prediction completed successfully!")

                st.subheader("🌍 Air Quality Prediction Result")

                st.metric(
                    "Predicted AQI Category",
                    str(prediction)
                )

                st.write("**City:**", city)

                st.write(
                    "**Prediction Date:**",
                    selected_date
                )

                st.write(
                    "**Model Used:** Decision Tree Classifier"
                )

                # Probability chart

                st.subheader("📊 Prediction Probability")

                fig, ax = plt.subplots(figsize=(9, 4))

                ax.barh(
                    probability_df["AQI Category"].astype(str),
                    probability_df["Probability (%)"]
                )

                ax.set_xlabel("Prediction Probability (%)")
                ax.set_xlim(0, 100)
                ax.set_title("AQI Category Prediction Analysis")

                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.subheader("📋 Input Measurements")

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------
# 9. SHAP AND LIME PAGE
# ------------------------------------------------------------

elif page == "SHAP & LIME":

    st.title("📊 Explainable AI")

    st.write("""
    Explainable AI helps us understand why the
    Machine Learning model produces a prediction.

    **SHAP:** Explains the contribution of features
    to the model prediction.

    **LIME:** Explains an individual prediction
    using a locally interpretable model.
    """)

    if "input_data" not in st.session_state:

        st.warning(
            "Please run a prediction first on the Predictor page."
        )

    else:

        input_data = st.session_state["input_data"]

        processed = preprocessor.transform(input_data)

        X_input = np.asarray(
            processed.toarray()
            if hasattr(processed, "toarray")
            else processed,
            dtype=float
        )

        # ----------------------------------------------------
        # SHAP EXPLANATION
        # ----------------------------------------------------

        st.subheader("SHAP Feature Contribution Analysis")

        try:

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(
                X_input
            )

            prediction = model.predict(processed)[0]

            class_index = list(model.classes_).index(
                prediction
            )

            if isinstance(shap_values, list):

                local_values = np.asarray(
                    shap_values[class_index]
                )[0]

            else:

                shap_values = np.asarray(shap_values)

                if shap_values.ndim == 3:

                    local_values = shap_values[
                        0, :, class_index
                    ]

                elif shap_values.ndim == 2:

                    local_values = shap_values[0]

                else:

                    raise ValueError(
                        "Unexpected SHAP output dimensions."
                    )

            shap_df = pd.DataFrame({

                "Feature": list(feature_names),

                "Importance": np.abs(local_values),

                "SHAP Value": local_values

            })

            shap_df = shap_df.sort_values(
                "Importance"
            ).tail(10)

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.barh(
                shap_df["Feature"],
                shap_df["SHAP Value"]
            )

            ax.set_title(
                "SHAP - Feature Contribution Analysis"
            )

            ax.set_xlabel("SHAP Value")

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:

            st.error(f"SHAP explanation failed: {e}")

        # ----------------------------------------------------
        # LIME EXPLANATION
        # ----------------------------------------------------

        st.subheader("LIME Local Prediction Explanation")

        try:

            lime_explainer = LimeTabularExplainer(

                training_data=np.asarray(
                    lime_training_data
                ),

                feature_names=list(feature_names),

                class_names=[
                    str(c) for c in model.classes_
                ],

                mode="classification",

                random_state=42

            )

            prediction = model.predict(processed)[0]

            class_index = list(model.classes_).index(
                prediction
            )

            explanation = lime_explainer.explain_instance(

                X_input[0],

                model.predict_proba,

                labels=(class_index,),

                num_features=10

            )

            lime_data = explanation.as_list(
                label=class_index
            )

            lime_df = pd.DataFrame(
                lime_data,
                columns=["Feature", "Contribution"]
            )

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.barh(
                lime_df["Feature"],
                lime_df["Contribution"]
            )

            ax.set_title(
                "LIME - Local Prediction Explanation"
            )

            ax.set_xlabel("Feature Contribution")

            plt.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:

            st.error(f"LIME explanation failed: {e}")

# ------------------------------------------------------------
# 10. INPUT GUIDE PAGE
# ------------------------------------------------------------

elif page == "Input Guide":

    st.title("📖 Input Guide")

    st.write(
        "The following information explains "
        "the input fields used by the prediction system."
    )

    with st.expander("City"):
        st.write("Enter the city name for air quality prediction.")

    with st.expander("PM2.5 Mean and Maximum"):
        st.write(
            "PM2.5 represents fine particulate matter "
            "in the air."
        )

    with st.expander("PM10 Mean and Maximum"):
        st.write(
            "PM10 represents particulate matter "
            "with a diameter of 10 micrometres or less."
        )

    with st.expander("NO2 Mean"):
        st.write("Nitrogen dioxide is an air pollutant.")

    with st.expander("SO2 Mean"):
        st.write("Sulphur dioxide is an air pollutant.")

    with st.expander("O3 Maximum"):
        st.write("Ozone is a pollutant that can form near ground level.")

    with st.expander("CO Maximum"):
        st.write("Carbon monoxide is a colourless gas.")

    with st.expander("Dust Mean"):
        st.write("Average dust measurement in the dataset.")

    with st.expander("AOD Mean"):
        st.write("Aerosol Optical Depth relates to airborne particles.")

    with st.expander("Hours Available"):
        st.write("Number of hours for which measurements are available.")

    with st.expander("Date"):
        st.write("Date of the air quality measurement.")

# ------------------------------------------------------------
# 11. FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#12345a;padding:15px;">
        <b>AGR STUDYORBIT</b><br>
        AI-Based Air Quality Management and Prediction System
    </div>
    """,
    unsafe_allow_html=True
)
