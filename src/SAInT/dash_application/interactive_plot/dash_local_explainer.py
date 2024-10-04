from dash import html
import numpy as np
from SAInT.dash_application.interactive_plot.local_explainer.local_lime_explainer import DashLocalLimeExplainer
from SAInT.dash_application.interactive_plot.local_explainer.local_shap_explainer import DashLocalShapExplainer
from SAInT.dash_application.interactive_plot.common import check_for_sensitive_features

class DashLocalExplainer:
    def __init__(self, application):
        """
        Initialize the DashLocalExplainer instance.

        :param application: The application instance.
        """
        self.application = application
        self.explain_lime = True
        self.explain_shap = True
        self.do_save = True
        self.local_lime_explainer = None
        self.local_shap_explainer = None

    def reset(self):
        self.explain_lime = True
        self.explain_shap = True
        self.do_save = True
        self.local_lime_explainer = None
        self.local_shap_explainer = None

    def _generate_explanation_body(self, sample_dict):
        """
        Generate the HTML body for the explanation popup.

        :param sample_dict: Dictionary containing sample data.

        :return: List of Dash HTML components.
        """
        y_value = sample_dict["y"]
        p_value = sample_dict["p"]
        body = [
            html.H6(sample_dict['output_name'])
        ]
        gt_pred_mae_text = f"groundtruth: {y_value:.5f}"
        if p_value is not None:
            gt_pred_mae_text += self._generate_prediction_info(y_value, p_value)
        body.extend([html.P(gt_pred_mae_text)])

        if p_value is not None:
            if self.explain_lime:
                body.append(self.local_lime_explainer.generate_info(sample_dict))
            if self.explain_shap:
                body.append(self.local_shap_explainer.generate_info(sample_dict))
        features = ", ".join([f"{k}: {v}" for k, v in sample_dict["x"].items()])
        body.extend([
            html.H6("Features"),
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

    def explain(self, sample_dict):
        """
        Create explanation for the sample and open it in a popup.

        :param sample_dict: Dictionary containing sample data.
        """
        body = self._generate_explanation_body(sample_dict)
        self.application.lsa_popup.set_content(body)
        self.application.lsa_popup.open()

    def create_explainer_instances(self):
        dls_train = self.application.trainer.dataloader.dls_train
        train_data = np.array(dls_train.xs.values, dtype=np.float32)
        if self.local_lime_explainer is None:
            self.local_lime_explainer = DashLocalLimeExplainer(application=self.application,
                                                               train_data=train_data,
                                                               do_save=self.do_save)
        if self.local_shap_explainer is None:
            self.local_shap_explainer = DashLocalShapExplainer(application=self.application,
                                                               train_data=dls_train,
                                                               do_save=self.do_save)

    def scan_and_filter_samples(self, explanation_type="lime"):
        """
        Scans and filters samples that contain sensitive features in their explanations.

        :param sample_list: List of sample dictionaries.
        :param explanation_type: "lime" or "shap" to specify which explainer to use.
        :return: List of samples that do not have sensitive features in the top features.
        """
        self.create_explainer_instances()

        filtered_samples = []
        input_names = self.application.trainer.dataloader.valid.input_names
        output_names = self.application.trainer.dataloader.valid.output_names
        num_outputs = len(output_names)
        dataframes = { "valid": self.application.trainer.dataloader.valid.dataframe,
                      "test": self.application.trainer.dataloader.test.dataframe }

        for dataset_name, df in dataframes.items():
            f = df[input_names]
            for sample_index, x in f.iterrows():
                for output_idx in range(0, num_outputs):
                    if explanation_type == "lime":
                        explanation = self.local_lime_explainer.explain(x, output_idx)
                        top_features = self.local_lime_explainer.explainer.get_top_n_features(explanation, top_n=5)
                    elif explanation_type == "shap":
                        explanation = self.local_shap_explainer.explain(x)
                        top_features = self.local_shap_explainer.explainer.get_top_n_features(explanation, top_n=5)
                    else:
                        raise ValueError("Invalid explanation type. Choose 'lime' or 'shap'.")
                    sensitive_features = check_for_sensitive_features(top_features)
                    # If no sensitive features found, add the sample to the filtered list
                    if not bool(sensitive_features):
                        filtered_samples.append((dataset_name, sample_index))

        return filtered_samples
