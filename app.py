# ============================================================
# AIR QUALITY MANAGEMENT AND PREDICTION SYSTEM
# Standalone Gradio application for GitHub deployment
# Built from the user's TRW01 (2).ipynb notebook
# ============================================================

from pathlib import Path
from datetime import date

import gradio as gr
import joblib
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ------------------------------------------------------------
# 1. LOAD THE SAVED MODEL AND PREPROCESSING OBJECTS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "air_quality_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file was not found. Expected location: "
        f"{MODEL_PATH}. Make sure the file is uploaded inside the models folder."
    )

pipeline = joblib.load(MODEL_PATH)

required_keys = ["model", "preprocessor", "feature_names", "lime_training_data"]
missing_keys = [key for key in required_keys if key not in pipeline]
if missing_keys:
    raise KeyError(
        "The uploaded pickle file is missing these saved items: "
        + ", ".join(missing_keys)
        + ". Re-save the model using notebook Cell 25, then upload it again."
    )

model = pipeline["model"]
preprocessor = pipeline["preprocessor"]
feature_names = list(pipeline["feature_names"])
lime_training_data = np.asarray(pipeline["lime_training_data"])

print("Air quality model loaded successfully.")
print("Model:", type(model).__name__)
print("Number of transformed features:", len(feature_names))
print("Classes:", model.classes_)

# ------------------------------------------------------------
# 2. PREDICTION FUNCTION (FROM NOTEBOOK CELL 30)
# ------------------------------------------------------------

def predict_aqi(
    city,
    pm25_mean,
    pm25_max,
    pm10_mean,
    pm10_max,
    no2_mean,
    so2_mean,
    o3_max,
    co_max,
    dust_mean,
    aod_mean,
    hours_available,
    selected_date
):
    if not city or not str(city).strip():
        raise gr.Error("Please enter a city name.")

    try:
        selected_date = pd.to_datetime(selected_date, errors="raise")

        input_data = pd.DataFrame([{
            "city": str(city).strip(),
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
            "day_of_week": selected_date.dayofweek
        }])

        processed = preprocessor.transform(input_data)
        prediction = model.predict(processed)[0]
        probabilities = model.predict_proba(processed)[0]

        probability_df = pd.DataFrame({
            "AQI Category": [str(value) for value in model.classes_],
            "Probability (%)": np.round(probabilities * 100, 2)
        })

        result = f"""
## 🌍 Air Quality Prediction Result

### Predicted AQI Category

# {prediction}

**Location:** {city}

**Prediction Date:** {selected_date.date()}

**Model Used:** {type(model).__name__}
"""
        return result, probability_df, input_data

    except Exception as exc:
        raise gr.Error(f"Prediction failed: {exc}") from exc


# ------------------------------------------------------------
# 3. PREDICTION PROBABILITY CHART (FROM NOTEBOOK CELL 31)
# ------------------------------------------------------------

def create_probability_chart(probability_df):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(
        probability_df["AQI Category"].astype(str),
        probability_df["Probability (%)"]
    )
    ax.set_xlabel("Prediction Probability (%)")
    ax.set_title("AQI Category Prediction Analysis")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 4. SHAP EXPLANATION (BASED ON NOTEBOOK CELL 32)
# ------------------------------------------------------------

def create_shap_chart(input_data):
    processed = preprocessor.transform(input_data)
    X_input = np.asarray(
        processed.toarray() if hasattr(processed, "toarray") else processed,
        dtype=float
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    predicted_class = model.predict(processed)[0]
    class_index = list(model.classes_).index(predicted_class)

    if isinstance(shap_values, list):
        local_values = np.asarray(shap_values[class_index])[0]
    else:
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            local_values = shap_values[0, :, class_index]
        elif shap_values.ndim == 2:
            local_values = shap_values[0]
        else:
            raise ValueError(
                f"Unexpected SHAP output dimensions: {shap_values.shape}"
            )

    names = list(feature_names)
    shap_df = pd.DataFrame({
        "Feature": names,
        "Importance": np.abs(local_values),
        "SHAP Value": local_values
    }).sort_values("Importance").tail(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
    ax.set_title("SHAP - Feature Contribution Analysis")
    ax.set_xlabel("SHAP Value")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 5. LIME EXPLANATION (BASED ON NOTEBOOK CELL 33)
# ------------------------------------------------------------

def create_lime_chart(input_data):
    processed = preprocessor.transform(input_data)
    X_input = np.asarray(
        processed.toarray() if hasattr(processed, "toarray") else processed,
        dtype=float
    )

    lime_explainer = LimeTabularExplainer(
        training_data=np.asarray(lime_training_data, dtype=float),
        feature_names=list(feature_names),
        class_names=[str(value) for value in model.classes_],
        mode="classification",
        random_state=42
    )

    prediction = model.predict(processed)[0]
    class_index = list(model.classes_).index(prediction)

    explanation = lime_explainer.explain_instance(
        X_input[0],
        model.predict_proba,
        labels=(class_index,),
        num_features=min(10, len(feature_names))
    )

    lime_data = explanation.as_list(label=class_index)
    lime_df = pd.DataFrame(
        lime_data,
        columns=["Feature", "Contribution"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(lime_df["Feature"], lime_df["Contribution"])
    ax.set_title("LIME - Local Prediction Explanation")
    ax.set_xlabel("Feature Contribution")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# 6. COMBINE PREDICTION, SHAP AND LIME (NOTEBOOK CELL 34)
# ------------------------------------------------------------

def generate_dashboard(
    city,
    pm25_mean,
    pm25_max,
    pm10_mean,
    pm10_max,
    no2_mean,
    so2_mean,
    o3_max,
    co_max,
    dust_mean,
    aod_mean,
    hours_available,
    selected_date
):
    result, probability_df, input_data = predict_aqi(
        city,
        pm25_mean,
        pm25_max,
        pm10_mean,
        pm10_max,
        no2_mean,
        so2_mean,
        o3_max,
        co_max,
        dust_mean,
        aod_mean,
        hours_available,
        selected_date
    )

    probability_fig = create_probability_chart(probability_df)

    try:
        shap_fig = create_shap_chart(input_data)
    except Exception as exc:
        shap_fig = None
        print("SHAP explanation could not be generated:", exc)

    try:
        lime_fig = create_lime_chart(input_data)
    except Exception as exc:
        lime_fig = None
        print("LIME explanation could not be generated:", exc)

    return (
        result,
        probability_fig,
        shap_fig,
        lime_fig,
        input_data
    )



# ============================================================
# AIR QUALITY MANAGEMENT AND PREDICTION SYSTEM
# MCA FINAL YEAR PROJECT
# AGR STUDYORBIT
# ============================================================

import gradio as gr
from datetime import date

# ------------------------------------------------------------
# 2. CSS - LIGHT BACKGROUND AND VISIBLE TEXT
# ------------------------------------------------------------

custom_css = """
/* Main application background */
.gradio-container {
    background: #f1f6fc !important;
    color: #172b4d !important;
}

/* Main page background */
#page-content {
    background: #f1f6fc !important;
    color: #172b4d !important;
    padding: 18px !important;
}

/* Sidebar */
#sidebar {
    background: #e1edfa !important;
    border: 1px solid #c6d9ee !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

/* Headings */
h1, h2, h3, h4,
#page-content label,
#sidebar label {
    color: #12345a !important;
}

/* Normal text */
#page-content p,
#page-content span,
#page-content li,
#page-content .prose,
#sidebar p {
    color: #243b53 !important;
}

/* Content panels and sections */
#page-content .block,
#page-content .form,
#page-content .panel,
#page-content .wrap,
#page-content .gr-box {
    background-color: #ffffff !important;
    color: #172b4d !important;
    border-color: #d6e2f0 !important;
}

/* Input fields */
#page-content input,
#page-content textarea,
#page-content select {
    background-color: #ffffff !important;
    color: #172b4d !important;
    border: 1px solid #b8cce2 !important;
}

/* Input placeholder */
#page-content input::placeholder,
#page-content textarea::placeholder {
    color: #718096 !important;
}

/* Buttons */
#sidebar button,
#page-content button {
    border-radius: 8px !important;
}

/* Dataframe background */
#input-data-table,
#input-data-table .table-wrap,
#input-data-table .table-container,
#input-data-table table,
#input-data-table thead,
#input-data-table tbody {
    background: #ffffff !important;
    color: #172b4d !important;
}

/* Dataframe headers */
#input-data-table th {
    background: #dceafa !important;
    color: #12345a !important;
    font-weight: bold !important;
}

/* Dataframe cells */
#input-data-table td {
    background: #ffffff !important;
    color: #172b4d !important;
    border-color: #d6e2f0 !important;
}

/* Dataframe alternating rows */
#input-data-table tr:nth-child(even) td {
    background: #f2f7fd !important;
}

/* Prediction result */
#result-box {
    background: #e7f2ff !important;
    border: 1px solid #b8d5f5 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Plot output areas */
#page-content .plot-container,
#page-content .plot,
#page-content .image-container {
    background: #ffffff !important;
    border-radius: 10px !important;
}

/* Footer */
#project-footer {
    background: #e1edfa !important;
    color: #12345a !important;
    text-align: center !important;
    padding: 15px !important;
    border-radius: 10px !important;
    font-weight: bold !important;
}

/* Accordion */
#page-content .label-wrap {
    color: #12345a !important;
}

/* Ensure text remains readable */
#page-content * {
    color-scheme: light;
}
"""

# ------------------------------------------------------------
# 3. PAGE NAVIGATION FUNCTION
# ------------------------------------------------------------

def show_page(page):
    return (
        gr.update(visible=(page == "home")),
        gr.update(visible=(page == "predictor")),
        gr.update(visible=(page == "explainability")),
        gr.update(visible=(page == "guide"))
    )

# ------------------------------------------------------------
# 4. CREATE GRADIO APPLICATION
# ------------------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="Air Quality Management And Prediction System"
) as demo:

    # --------------------------------------------------------
    # APPLICATION HEADER
    # --------------------------------------------------------



    # APPLICATION HEADER

    gr.Markdown(
        """
        <div style="
            background: linear-gradient(90deg, #12345a, #2878bd);
            padding: 22px;
            border-radius: 14px;
            text-align: center;
        ">
            <h1 style="color: orange !important;">
                🌍 AIR QUALITY MANAGEMENT SYSTEM
            </h1>
            <h3 style="color: white !important;">
                AI-Based Air Quality Management and Prediction
                System Using Machine Learning
            </h3>
            <p style="color: white !important;">
            </p>
        </div>
        """
    )

    gr.Markdown(" ")




    gr.Markdown(" ")

    # --------------------------------------------------------
    # MAIN LAYOUT: SIDEBAR + CONTENT
    # --------------------------------------------------------

    with gr.Row():

        # SIDEBAR
        with gr.Column(scale=1, min_width=190, elem_id="sidebar"):

            gr.Markdown("## 📌 Navigation")

            home_nav = gr.Button("🏠 Home", variant="primary")
            predictor_nav = gr.Button("🔍 Predictor")
            explain_nav = gr.Button("📊 SHAP & LIME")
            guide_nav = gr.Button("📖 Input Guide")

            gr.Markdown("---")

            gr.Markdown(
                """
                **Project Information**

                Domain: Big Data + Machine Learning

                Platform: Python / Google Colab

                Interface: Gradio
                """
            )

        # MAIN CONTENT
        with gr.Column(scale=4, elem_id="page-content"):

            # =================================================
            # PAGE 1: HOME
            # =================================================

            with gr.Column(visible=True) as home_page:

                gr.Markdown("# 🏠 Home")

                gr.Markdown(
                    """
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
                    """
                )

                gr.Markdown("### ⚙️ Project Workflow")

                gr.Markdown(
                    """
                    1. Dataset collection
                    2. Data cleaning and preprocessing
                    3. Exploratory data analysis
                    4. Machine Learning model training
                    5. Model evaluation
                    6. Air quality prediction
                    7. Explainable AI using SHAP and LIME
                    """
                )

                start_button = gr.Button(
                    "🚀 Start Air Quality Prediction",
                    variant="primary",
                    size="lg"
                )

            # =================================================
            # PAGE 2: PREDICTOR
            # =================================================

            with gr.Column(visible=False) as predictor_page:

                gr.Markdown("# 🔍 Air Quality Predictor")

                gr.Markdown(
                    "Enter the air quality measurements below "
                    "and click Predict."
                )

                with gr.Row():

                    with gr.Column():

                        city = gr.Textbox(
                            label="City",
                            value="Delhi",
                            placeholder="Enter city name"
                        )

                        selected_date = gr.Textbox(
                            label="Date (YYYY-MM-DD)",
                            value=date.today().isoformat()
                        )

                        pm25_mean = gr.Number(
                            label="PM2.5 Mean",
                            value=80
                        )

                        pm25_max = gr.Number(
                            label="PM2.5 Maximum",
                            value=120
                        )

                        pm10_mean = gr.Number(
                            label="PM10 Mean",
                            value=150
                        )

                        pm10_max = gr.Number(
                            label="PM10 Maximum",
                            value=200
                        )

                        no2_mean = gr.Number(
                            label="NO2 Mean",
                            value=40
                        )

                    with gr.Column():

                        so2_mean = gr.Number(
                            label="SO2 Mean",
                            value=15
                        )

                        o3_max = gr.Number(
                            label="O3 Maximum",
                            value=50
                        )

                        co_max = gr.Number(
                            label="CO Maximum",
                            value=1.2
                        )

                        dust_mean = gr.Number(
                            label="Dust Mean",
                            value=80
                        )

                        aod_mean = gr.Number(
                            label="AOD Mean",
                            value=0.5
                        )

                        hours_available = gr.Slider(
                            minimum=0,
                            maximum=24,
                            value=24,
                            step=1,
                            label="Hours Available"
                        )

                predict_button = gr.Button(
                    "🔮 Predict Air Quality",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("## 📌 Prediction Result")

                prediction_output = gr.Textbox(
                    label="Predicted Air Quality Category",
                    lines=3,
                    interactive=False,
                    elem_id="result-box"
                )

                probability_output = gr.Plot(
                    label="Prediction Probability"
                )

                gr.Markdown("## 📋 Input Measurements")

                input_output = gr.Dataframe(
                    label="Air Quality Measurements",
                    interactive=False,
                    elem_id="input-data-table",
                    wrap=True
                )

            # =================================================
            # PAGE 3: SHAP AND LIME
            # =================================================

            with gr.Column(visible=False) as explainability_page:

                gr.Markdown("# 📊 Explainable AI")

                gr.Markdown(
                    """
                    Explainable AI helps us understand why the
                    Machine Learning model produces a prediction.

                    **SHAP:** Explains the contribution of features
                    to the model prediction.

                    **LIME:** Explains an individual prediction
                    using a locally interpretable model.
                    """
                )

                gr.Markdown("## SHAP Explanation")

                shap_output = gr.Plot(
                    label="SHAP Feature Importance"
                )

                gr.Markdown("## LIME Explanation")

                lime_output = gr.Plot(
                    label="LIME Explanation"
                )

                gr.Markdown(
                    """
                    **Note:** First run a prediction on the Predictor
                    page to generate the explanation plots.
                    """
                )

            # =================================================
            # PAGE 4: INPUT GUIDE
            # =================================================

            with gr.Column(visible=False) as guide_page:

                gr.Markdown("# 📖 Input Guide")

                gr.Markdown(
                    "The following table explains the input fields "
                    "used by the prediction system."
                )

                with gr.Accordion("City", open=False):
                    gr.Markdown(
                        "Enter the name of the city for which "
                        "you want to predict air quality."
                    )

                with gr.Accordion("PM2.5 Mean and Maximum", open=False):
                    gr.Markdown(
                        "PM2.5 represents fine particulate matter "
                        "in the air. Mean and maximum are its "
                        "average and highest recorded values."
                    )

                with gr.Accordion("PM10 Mean and Maximum", open=False):
                    gr.Markdown(
                        "PM10 represents particulate matter with "
                        "a diameter of 10 micrometres or less."
                    )

                with gr.Accordion("NO2 Mean", open=False):
                    gr.Markdown(
                        "Nitrogen dioxide is an air pollutant "
                        "commonly associated with combustion "
                        "and vehicle emissions."
                    )

                with gr.Accordion("SO2 Mean", open=False):
                    gr.Markdown(
                        "Sulphur dioxide is a gas released mainly "
                        "from burning sulphur-containing fuels."
                    )

                with gr.Accordion("O3 Maximum", open=False):
                    gr.Markdown(
                        "Ozone is a pollutant that can form near "
                        "ground level through chemical reactions."
                    )

                with gr.Accordion("CO Maximum", open=False):
                    gr.Markdown(
                        "Carbon monoxide is a colourless gas "
                        "produced by incomplete combustion."
                    )

                with gr.Accordion("Dust Mean", open=False):
                    gr.Markdown(
                        "Average dust measurement in the dataset."
                    )

                with gr.Accordion("AOD Mean", open=False):
                    gr.Markdown(
                        "Aerosol Optical Depth is a measure related "
                        "to the amount of airborne particles."
                    )

                with gr.Accordion("Hours Available", open=False):
                    gr.Markdown(
                        "Number of hours for which measurements "
                        "are available."
                    )

                with gr.Accordion("Date", open=False):
                    gr.Markdown(
                        "Enter the measurement date in "
                        "YYYY-MM-DD format."
                    )

            # ------------------------------------------------
            # FOOTER
            # ------------------------------------------------

            gr.Markdown(
                """
                <div id="project-footer">
                    AGR STUDYORBIT                    <br>
                    AI-Based Air Quality Management and Prediction System
                </div>
                """
            )

    # --------------------------------------------------------
    # 5. SIDEBAR NAVIGATION EVENTS
    # --------------------------------------------------------

    page_outputs = [
        home_page,
        predictor_page,
        explainability_page,
        guide_page
    ]

    home_nav.click(
        fn=lambda: show_page("home"),
        outputs=page_outputs
    )

    predictor_nav.click(
        fn=lambda: show_page("predictor"),
        outputs=page_outputs
    )

    explain_nav.click(
        fn=lambda: show_page("explainability"),
        outputs=page_outputs
    )

    guide_nav.click(
        fn=lambda: show_page("guide"),
        outputs=page_outputs
    )

    start_button.click(
        fn=lambda: show_page("predictor"),
        outputs=page_outputs
    )

    # --------------------------------------------------------
    # 6. CONNECT YOUR EXISTING PREDICTION FUNCTION
    # --------------------------------------------------------

    predict_button.click(
        fn=generate_dashboard,

        inputs=[
            city,
            pm25_mean,
            pm25_max,
            pm10_mean,
            pm10_max,
            no2_mean,
            so2_mean,
            o3_max,
            co_max,
            dust_mean,
            aod_mean,
            hours_available,
            selected_date
        ],

        outputs=[
            prediction_output,
            probability_output,
            shap_output,
            lime_output,
            input_output
        ]
    )

# ------------------------------------------------------------
# 7. LAUNCH APPLICATION
# ------------------------------------------------------------

demo.launch(server_name="0.0.0.0", server_port=7860)
