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
