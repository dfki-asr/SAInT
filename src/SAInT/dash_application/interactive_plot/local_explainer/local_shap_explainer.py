from dash import html
from timeit import default_timer as timer
from SAInT.sa.lsa_shap import LocalShapExplainer
from SAInT.dash_application.interactive_plot.common import scale_html

class DashLocalShapExplainer:
    def __init__(self, application, train_data, do_save):
        pixel_def = application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        self.shap_height = pixel_def.shap_height
        self.nsamples = 100
        self.explainer = LocalShapExplainer(
            model=application.model_handler.best_model,
            data=train_data,
            nsamples=self.nsamples,
            figure_folder=application.trainer.figure_folder
        )
        self.colors = application.color_palette.to_hex_list()
        self.do_save = do_save

    def explain(self, x):
        if self.explainer is None:
            raise RuntimeError("Local SHAP Explainer is not initialized!")
        start = timer()
        explanation = self.explainer.explain(x)
        print(f"LSA with SHAP took {(timer() - start):.2f} s.")
        return explanation

    def _create_html(self, explanation, sample_dict):
        best_model_name = self.explainer.model.name
        shap_html = self.explainer.plot(
            explanation=explanation,
            title=f"{best_model_name}_SHAP_n{self.nsamples}",
            output_idx=sample_dict["output_idx"],
            colors=self.colors,
            do_save=self.do_save
        )
        return scale_html(shap_html, scale=0.9)

    def generate_info(self, sample_dict):
        explanation = self.explain(x=sample_dict["x"])
        src_shap = self._create_html(explanation, sample_dict)
        return html.Div([
            html.H6("LSA with SHAP"),
            html.Iframe(srcDoc=src_shap, height=self.shap_height, width="100%")
        ])
