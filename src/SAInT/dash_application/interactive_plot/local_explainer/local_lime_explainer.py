from dash import html
from timeit import default_timer as timer
from SAInT.sa.lsa_lime import LocalLimeExplainer
from SAInT.dash_application.interactive_plot.common import scale_html

class DashLocalLimeExplainer:
    def __init__(self, application, train_data, do_save):
        pixel_def = application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        lime_expl_width = pixel_def.lime_expl_width
        self.lime_height = pixel_def.lime_height
        self.explainer = LocalLimeExplainer(
            model=application.model_handler.best_model,
            data=train_data,
            data_type="tabular",
            figure_folder=application.trainer.figure_folder,
            lime_expl_width=lime_expl_width
        )
        self.colors = application.color_palette.to_decimal_list()
        self.do_save = do_save

    def explain(self, x, output_idx):
        if self.explainer is None:
            raise RuntimeError("Local LIME Explainer is not initialized!")
        start = timer()
        num_features = min(15, x.shape[-1])
        explanation = self.explainer.explain(x,
                                        num_features=num_features,
                                        output_idx=output_idx)
        print(f"LSA with LIME took {(timer() - start):.2f} s.")
        return explanation

    def _create_html(self, explanation):
        best_model_name = self.explainer.model.name
        lime_html = self.explainer.plot(
            explanation=explanation,
            title=f"{best_model_name}_LIME",
            colors=self.colors,
            do_show=False,
            do_save=self.do_save
        )
        return scale_html(lime_html, scale=0.9)

    def generate_info(self, sample_dict):
        explanation = self.explain(x=sample_dict["x"],
                                   output_idx=sample_dict["output_idx"])
        src_lime = self._create_html(explanation)
        return html.Div([
            html.H6("LSA with LIME"),
            html.Iframe(srcDoc=src_lime, height=self.lime_height, width="100%")
        ])
