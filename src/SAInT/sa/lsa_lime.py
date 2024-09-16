from typing import Union
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from SAInT.model import Model
from SAInT.common import makedirs


class LocalLimeExplainer():

    def __init__(self,
                 model: Model,
                 data: np.array,
                 data_type: str = "tabular",
                 figure_folder: str = "",
                 lime_expl_width: str = "65%"):
        self.model = model
        self.data = data
        self.type = data_type
        self.explainer = None
        self.figure_folder = figure_folder
        self.lime_expl_width = lime_expl_width

        if len(self.model.categorical_cols) > 0:
            raise RuntimeError("Error: Dataset has categorial features!")

        if isinstance(data, np.ndarray):
            feature_names=self.model.continuous_cols + self.model.categorical_cols
            data = pd.DataFrame(data, columns=feature_names)

        # Check if there are NaN values in the data
        if np.any(np.isnan(data)):
            all_nan_columns = data.columns[data.isna().all()].tolist()
            if len(all_nan_columns) > 0:
                raise ValueError(f"LIME error: all values are NaN in {all_nan_columns}")
            mean_values = data.mean(skipna=True)
            if np.any(np.isnan(mean_values)):
                mean_nan_columns = mean_values.columns[mean_values.isna().all()].tolist()
                raise ValueError(f"LIME error: Mean is NaN: {mean_nan_columns}")
            # otherwise, fill those nan samples in data with mean
            data = data.fillna(mean_values)

        # Check if there are still NaN values in the data
        still_nan_columns = data.columns[data.isna().any()].tolist()
        if len(still_nan_columns) > 0:
            raise ValueError(f"LIME error: Still NaN values in data! Columns with NaNs: {still_nan_columns}")

        if self.type == "tabular":
            if model.mode == "regression":
                class_names = model.target_names
            if model.mode == "classification":
                class_names = []
                for t in model.target_names:
                    class_names.append(f"{t}=0")
                    class_names.append(f"{t}=1")
            self.explainer = LimeTabularExplainer(
                data.values,
                feature_names=model.continuous_cols + model.categorical_cols,
                class_names=class_names,
                categorical_features=model.categorical_cols,
                verbose=model.verbose,
                mode=model.mode,
                discretize_continuous=True)
        else:
            raise RuntimeError("Other data types not yet supported!")

    def predict_output_at_index(self, sample: Union[pd.DataFrame, np.ndarray], output_idx: int = 0) -> np.ndarray:
        # Get the prediction list from the model
        predictions = self.model.predict(sample)
        mode = self.model.mode
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            if mode == "regression":
                return predictions
            elif mode == "classification":
                # For binary classification, we need both class probabilities
                probabilities = self.model.predict_proba(sample)
                return probabilities
            else:
                raise RuntimeError(f"Unknown mode {mode}!")
        output = predictions[:,output_idx].reshape(-1, 1)
        return output

    def explain(self, sample: np.ndarray, num_features: int = 5, output_idx: int = 0):
        sample_squeezed = sample.squeeze()
        # Check if NaN value in sample
        if np.any(np.isnan(sample_squeezed)):
            raise ValueError("LIME error: NaN values in sample!")

        num_neighbor_samples = 5000
        explanation = self.explainer.explain_instance(
            data_row=sample_squeezed,
            predict_fn=lambda x: self.predict_output_at_index(x, output_idx=output_idx),
            num_samples=num_neighbor_samples,
            num_features=num_features
        )

        return explanation

    def get_top_n_features(self, explanation, top_n: int = 5):
        """ This function returns a list of the top_n most important features, either positively or negatively.
        Parameters:
        - explanation: the LIME explanation object
        - top_n: the number of top features to consider (default is 5)
        Returns:
        - List of the top n most important features.
        """
        # Get the list of features and their weights from LIME explanation
        feature_importances = explanation.as_list()
        # Sort features by absolute weight (importance)
        sorted_features = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
        # Extract the top_n most important feature names
        top_features = [name for name, _ in sorted_features[:top_n]]
        return top_features

    def plot(self, explanation, title: str, colors: list = ["blue", "orange"], do_show = False, do_save = False):
        kwargs = {
            "predict_proba": True,
            "show_predicted_value": True,
            "show_table": True
        }
        if do_show:
            explanation.show_in_notebook(**kwargs)
        lime_html = explanation.as_html(**kwargs)

        # Increase width of explanation "positive"/"negative" figure
        lime_expl_size_replace_str = ".lime.explanation { width: " + self.lime_expl_width + "; }"
        lime_html = str(lime_html).replace(
            ".lime.explanation {\\n  width: 350px;\\n}\\n\\n",
            lime_expl_size_replace_str
        )
        # Move table closer to explanation "positive"/"negative" figure
        lime_html = lime_html.replace(
		    ".lime.text_div {\\n  max-height:300px;\\n  flex: 1 0 300px;\\n  overflow:scroll;\\n}\\n",
            ""
        )
        lime_html = lime_html.replace(
		    ".lime.table_div {\\n  max-height:300px;\\n  flex: 1 0 300px;\\n  overflow:scroll;\\n}\\n",
            ""
        )
        # replace colors
        lime_html = lime_html.replace(
		    "var d3_category10 = [ 2062260, 16744206, 2924588, 14034728, 9725885, 9197131, 14907330, 8355711, 12369186, 1556175 ].map(d3_rgbString);",
            f"var d3_category10 = [ {colors[0]}, {colors[1]}, {colors[2]}, {colors[3]}, {colors[4]}, 9197131, 14907330, 8355711, 12369186, 1556175 ].map(d3_rgbString);"
        )
        lime_html = lime_html.replace(
		    "var d3_category20 = [ 2062260, 11454440, 16744206, 16759672, 2924588, 10018698, 14034728, 16750742, 9725885, 12955861, 9197131, 12885140, 14907330, 16234194, 8355711, 13092807, 12369186, 14408589, 1556175, 10410725 ].map(d3_rgbString);",
            f"var d3_category20 = [ {colors[0]}, {colors[1]}, {colors[2]}, {colors[3]}, {colors[4]}, 10018698, 14034728, 16750742, 9725885, 12955861, 9197131, 12885140, 14907330, 16234194, 8355711, 13092807, 12369186, 14408589, 1556175, 10410725 ].map(d3_rgbString);"
        )
        if do_save:
            makedirs(self.figure_folder)
            figure_path = f"{self.figure_folder}/{title}"
            with open(f"{figure_path}.html", "w", encoding="utf-8") as file:
                file.write(lime_html)

        return lime_html
