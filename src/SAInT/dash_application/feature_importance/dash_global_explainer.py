from timeit import default_timer as timer
from SAInT.sa.gsa import GlobalExplainer
from SAInT.dash_application.common.image_loader import ImageLoader


class DashGlobalExplainer:
    def __init__(self, application):
        """
        Initialize the DashGlobalExplainer instance.

        :param application: The application instance.
        """
        self.application = application
        self.image_loader = ImageLoader()

    def _perform_gsa_analysis(self, method: str = "fast", colors: list = ["blue", "orange"], do_save: bool = True):
        """
        Perform GSA analysis and update the application with the results.

        :param method: The method used for GSA ('fast' by default).
        :param colors: List of colors used for the GSA plot.
        :param do_save: Flag to save the GSA figure.
        :return: The GSA figure.
        """
        start_time = timer()
        global_explainer = GlobalExplainer(model=self.application.model_handler.best_model,
                                           num_samples=self.application.settings.num_samples,
                                           data=self.application.trainer.dataloader.dls_train,
                                           do_show=False, do_save=do_save,
                                           figure_folder=self.application.trainer.figure_folder)
        top_n_features, gsa_figure = global_explainer.apply_gsa(method=method,
                                                                num_top_features=self.application.settings.num_top_features,
                                                                colors=colors)
        self.application.most_important_features = top_n_features["total"]
        elapsed_time = timer() - start_time
        print(f"GSA took {elapsed_time:.2f} seconds.")
        return gsa_figure

    def explain(self, method: str = "fast", colors: list = ["blue", "orange"], do_save: bool = False, width: str = None, height: str = None):
        """
        Explain GSA by performing the analysis and loading the figure.

        :param method: The method used for GSA ('fast' by default).
        :param colors: List of colors used for the GSA plot.
        :param do_save: Flag to save the GSA figure.
        :param width: The width of the figure.
        :param height: The height of the figure.
        :return: The source of the loaded SVG.
        """
        if self.application.model_handler.best_model is None:
            print("Identify the best model first!")
            return None

        self.application.most_important_features = None
        gsa_figure = self._perform_gsa_analysis(method=method, colors=colors, do_save=do_save)
        src = self.image_loader.load_svg_from_plt(gsa_figure, width=width, height=height, title="GSA")
        return src
