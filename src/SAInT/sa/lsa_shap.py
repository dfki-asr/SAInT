import numpy as np
import shap
from SAInT.model import Model
from collections.abc import Iterable
from SAInT.common import makedirs


class LocalShapExplainer():

    def __init__(self,
                 model: Model,
                 data: np.array,
                 nsamples: int = 100,
                 figure_folder: str = ""):
        self.model = model
        self.figure_folder = figure_folder
        all_data_samples = data.xs.iloc[:]
        if nsamples > len(all_data_samples):
            raise RuntimeError(
                f"nsamples ({nsamples}) exceeds limit {len(all_data_samples)}!"
            )
        self.background_data = shap.sample(all_data_samples, nsamples)

        self.explainer = shap.KernelExplainer(
            self.model.predict, self.background_data)


    def explain(self, inputs: np.ndarray):
        explanation = self.explainer(inputs)
        return explanation

    def plot(self, explanation, title: str, output_idx: int = 0, colors: list = ["blue", "orange"], do_save=False):
        shap.initjs()
        total_base_values = self.explainer.expected_value
        total_shap_values = explanation.values
        if isinstance(total_base_values, Iterable):
            base_value = total_base_values[output_idx]
            shap_values = total_shap_values[:, output_idx]
        else:
            base_value = total_base_values
            shap_values = total_shap_values

        feature_names = list(explanation.data.keys())
        shap_fig = shap.force_plot(base_value,
                                   shap_values,
                                   feature_names=feature_names,
                                   show=False,
                                   matplotlib=False)

        shap_html = f"<head>{shap.getjs()}</head><body>{shap_fig.html()}</body>"
        # replace colors
        shap_html = shap_html.replace(
		    "const je={colors:{RdBu:[\"rgb(255, 13, 87)\",\"rgb(30, 136, 229)\"],",
		    f'const je={{colors:{{RdBu:[\"{colors[1]}\",\"{colors[0]}\"],'
        )
        if do_save:
            makedirs(self.figure_folder)
            figure_path = f"{self.figure_folder}/{title}_force_plot"
            with open(f"{figure_path}.html", "w", encoding="utf-8") as file:
                file.write(shap_html)
        return shap_html
