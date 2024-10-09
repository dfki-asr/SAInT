from SAInT.dash_application.settings.app_settings import ApplicationSettings
from SAInT.dash_application.colors.color_palette_registry import ColorPaletteRegistry
from SAInT.dash_application.interactive_plot.interactive_plot import InteractivePlot
from SAInT.dash_application.interactive_plot.dash_local_explainer import DashLocalExplainer
from SAInT.dash_application.model_definition.model_handler import ModelHandler
from SAInT.dash_application.components.dash_popup import DashPopup
from SAInT.dash_application.pixel_definitions import PixelDefinitions

class SAInTApplication:

    def __init__(self, settings: ApplicationSettings = None):
        self.pixel_definitions = PixelDefinitions(width=1386, height=768)
        palettes = ColorPaletteRegistry()
        palettes.setup()
        self.color_palette = palettes.palette_dict["saint"]
        self.trainer = None
        self.settings = settings
        self.most_important_features = None
        self.lsa_popup = DashPopup(
            title="Local Sensitivity Analysis (LSA)",
            id_popup="lsa_popup",
            id_window="lsa_window_popup_div",
            id_close="close_lsa_popup"
        )
        self.feature_selection_popup = DashPopup(
            title="Feature Selection",
            id_popup="feature_selection_popup",
            id_window="outputnames_window_popup_div",
            id_close="close_feature_selection_popup",
            id_save="save_outputnames_button",
            fullscreen=False
        )
        self.gsa_figure = None
        self.error_figure = None
        self.show_datasets_in_plot = ["train"]
        self.interactive_plot = InteractivePlot(application=self)
        self.local_explainer = DashLocalExplainer(application=self)
        self.model_handler = ModelHandler(application=self)

    @property
    def most_important_features_text(self):
        if self.most_important_features is not None:
            features_str = ", ".join(self.most_important_features)
            best_model_text = f"Best model: '{self.model_handler.best_model.path}'."
            error_text = f"Error measures:\n{self.model_handler.std_error_str}"
            most_important_features_text = f"The most important features (highest Total order Sobol) are:\n{features_str}."
            gsa_explanation = "Note:\n" \
            +"The higher the First order Sobol, the higher the direct influence of the feature onto the model prediction (not considering interactions with other features).\n" \
            +"The higher the Total order Sobol, the higher the influence of the feature onto the prediction including interactions with other features.\n" \
            +"The difference between the First order Sobol and Total order Sobol indicates the amount of interdependencies of the feature with other features.\n" \
            +"X axis: Feature, Y axis: Sobol Value (0=0%, 1=100%)\n" \
            +"Black vertical lines: Confidence Interval.\n"
            return f"{best_model_text}\n\n{error_text}\n{most_important_features_text}\n\n{gsa_explanation}"
        return ""

    @property
    def data_folder(self):
        if self.trainer is not None:
            return self.trainer.data_folder
        return "no data folder selected."

    def reset(self):
        if self.trainer is not None:
            self.trainer.models.clear()
            self.trainer.errors.clear()
        self.model_handler.best_model = None
        self.model_handler.std_error_str = ""
        self.most_important_features = None
        self.gsa_figure = None
        self.error_figure = None
        self.lsa_popup.reset()
        # reset data points
        self.interactive_plot.reset()
        self.local_explainer.reset()
        self.local_explainer.reset()
