from dash import html
import dash_bootstrap_components as dbc
from timeit import default_timer as timer
from SAInT.sa.lsa_lime import LocalLimeExplainer
from SAInT.sa.lsa_shap import LocalShapExplainer
from SAInT.dash_application.components import DashRadioButton

class DashLocalExplainer:
    def __init__(self, application):
        """
        Initialize the DashLocalExplainer instance.

        :param application: The application instance.
        """
        self.application = application
        self.hex_colors = application.color_palette.to_hex_list()
        self.decimal_colors = application.color_palette.to_decimal_list()
        self.explain_lime = True
        self.explain_shap = True
        self.do_save = True

    def _explain_local_lime(self, sample_dict):
        """
        Generate and return the HTML representation of the Lime explanation.

        :param sample_dict: Dictionary containing sample data.

        :return: HTML representation of the Lime explanation.
        """
        start = timer()
        train_data = sample_dict["train_data"]
        explainer = LocalLimeExplainer(
            model=self.application.model_handler.best_model,
            data=train_data,
            data_type="tabular",
            figure_folder=self.application.trainer.figure_folder
        )
        num_features = min(15, sample_dict["x"].shape[-1])
        explanation = explainer.explain(sample_dict["x"],
                                        num_features=num_features,
                                        output_idx=sample_dict["output_idx"])
        print(f"LSA with LIME took {(timer() - start):.2f} s.")

        title = f"{self.application.model_handler.best_model.name}_LIME"
        lime_html = explainer.plot(
            explanation=explanation,
            title=title,
            colors=self.decimal_colors,
            do_show=False,
            do_save=self.do_save
        )
        lime_html = self._scale_html(lime_html, max_width=1800)
        return lime_html

    def _explain_local_shap(self, sample_dict):
        """
        Generate and return the HTML representation of the SHAP explanation.

        :param sample_dict: Dictionary containing sample data.

        :return: HTML representation of the SHAP explanation.
        """
        start = timer()
        shap_explainer = LocalShapExplainer(
            model=self.application.model_handler.best_model,
            data=sample_dict["dls_train"],
            nsamples=100,
            figure_folder=self.application.trainer.figure_folder
        )
        explanation = shap_explainer.explain(sample_dict["x"])
        print(f"LSA with SHAP took {(timer() - start):.2f} s.")

        title = f"{self.application.model_handler.best_model.name}_SHAP_n100"
        shap_html = shap_explainer.plot(
            explanation=explanation,
            title=title,
            output_idx=sample_dict["output_idx"],
            colors=self.hex_colors,
            do_save=self.do_save
        )
        shap_html = self._scale_html(shap_html)
        return shap_html

    def _generate_explanation_body(self, sample_dict):
        """
        Generate the HTML body for the explanation popup.

        :param sample_dict: Dictionary containing sample data.

        :return: List of Dash HTML components.
        """
        y_value = sample_dict["y"]
        p_value = sample_dict["p"]
        body = [
            html.H3(sample_dict['output_name'])
        ]
        gt_pred_mae_text = f"groundtruth: {y_value:.5f}"
        if p_value is not None:
            gt_pred_mae_text += self._generate_prediction_info(y_value, p_value)
        body.extend([html.P(gt_pred_mae_text)])

        if p_value is not None:
            if self._should_explain_lime():
                body.append(self._generate_lime_info(sample_dict))
            if self._should_explain_shap():
                body.append(self._generate_shap_info(sample_dict))
        features = ", ".join([f"{k}: {v}" for k, v in sample_dict["x"].items()])
        body.extend([
            html.H3("Features"),
            html.P(features)
        ])
        return body

    def _generate_prediction_info(self, y_value, p_value):
        """
        Generate the prediction information HTML components.

        :param y_value: The ground truth value.
        :param p_value: The predicted value.

        :return: List of Dash HTML components containing prediction info.
        """
        mae_err = abs(y_value - p_value)
        mae_err_percent = f"({(mae_err / y_value) * 100.0:.2f}%)" if y_value != 0.0 else ""
        return f",   prediction: {p_value:.5f},   MAE: {mae_err:.5f} {mae_err_percent}"

    def _should_explain_lime(self):
        """
        Determine if Lime explanation should be generated.

        :return: Boolean indicating if Lime explanation should be generated.
        """
        return self.explain_lime

    def _should_explain_shap(self):
        """
        Determine if SHAP explanation should be generated.

        :return: Boolean indicating if SHAP explanation should be generated.
        """
        return self.explain_shap

    def _generate_lime_info(self, sample_dict):
        """
        Generate the Lime explanation HTML component.

        :param sample_dict: Dictionary containing sample data.

        :return: Dash HTML Div component with Lime explanation.
        """
        src_lime = self._explain_local_lime(sample_dict)
        return html.Div([
            html.H3("LSA with LIME"),
            html.Iframe(srcDoc=src_lime, height="550px", width="100%")
        ])

    def _generate_shap_info(self, sample_dict):
        """
        Generate the SHAP explanation HTML component.

        :param sample_dict: Dictionary containing sample data.

        :return: Dash HTML Div component with SHAP explanation.
        """
        src_shap = self._explain_local_shap(sample_dict)
        return html.Div([
            html.H3("LSA with SHAP"),
            html.Iframe(srcDoc=src_shap, height="350px", width="100%")
        ])

    def _scale_html(self, html_content, scale=2.0, max_width=None):
        """
        Scale the HTML content for better display.

        :param html_content: The HTML content to be scaled.
        :param scale: The scaling factor.
        :param max_width: The maximum width for the scaled content.

        :return: The scaled HTML content.
        """
        style = f"transform: scale({scale}); transform-origin: top left;"
        if max_width:
            style += f" max-width: {max_width}px; overflow-x: hidden;"
        return html_content.replace("<body>", f"<body style=\"{style}\">")

    def explain(self, sample_dict):
        """
        Create explanation for the sample and open it in a popup.

        :param sample_dict: Dictionary containing sample data.
        """
        body = self._generate_explanation_body(sample_dict)
        self.application.lsa_popup.set_content(body)
        self.application.lsa_popup.open()
