from timeit import default_timer as timer
from SAInT.dash_application.common.path_functions import get_file_extension, infer_modelfolder_from_modelfilename
from SAInT.dash_application.common.data_dialog import ask_for_directory, ask_for_file

class ModelHandler:
    def __init__(self, application):
        self.application = application
        self.std_error_str = ""
        self.best_model_was_changed = False
        self.best_model = None

    def _get_best_model(self, dataset_selection, loss_selection):
        trainer = self.application.trainer
        if len(trainer.errors[dataset_selection]) > 0:
            if len(trainer.errors[dataset_selection][loss_selection]) > 0:
                error_dict = trainer.errors[dataset_selection][loss_selection]
                self.best_model = trainer.get_best_model(error_dict=error_dict)

    def load_models(self):
        """Load models from the selected folder."""
        if self.application.trainer is not None:
            title = "Select Model Folder"
            model_folder = ask_for_directory(
                data_dir=self.application.trainer.model_folder,
                title=title
            )
            if not model_folder:
                return
            self.application.trainer.model_folder = model_folder

            start = timer()
            self.application.trainer.load_models()
            print(f"Load models took {(timer() - start):.2f} s.")

    def load_model(self):
        """Load a specific model from a selected .pkl file."""
        if self.application.trainer is not None:
            title = "Select model.pkl File"
            default_model_folder = self.application.trainer.model_folder
            model_filename = ask_for_file(
                data_dir=default_model_folder,
                title=title
            )
            if not model_filename:
                return
            if get_file_extension(model_filename) != ".pkl":
                print(f"Invalid model file '{model_filename}'! Load '.pkl' file instead!")
                return
            model_folder = infer_modelfolder_from_modelfilename(model_filename)
            print(f"Model folder = {model_folder}")
            self.application.trainer.model_folder = default_model_folder

            start = timer()
            self.application.trainer.load_model(model_filename)
            print(f"Load model took {(timer() - start):.2f} s.")

    def add_model(self, selected_type, base_model_name, settings_dict):
        """Train a new model."""
        colors = self.application.color_palette.to_normalized_rgba_list()
        if selected_type == "rf_model_selected":
            max_depth=settings_dict["max_depth"]
            n_estimators=settings_dict["n_estimators"]
            self.application.trainer.train_rf(base_model_name=base_model_name,
                                                max_depth=max_depth, n_estimators=n_estimators)
        elif selected_type == "xgb_model_selected":
            max_depth=settings_dict["max_depth"]
            n_estimators=settings_dict["n_estimators"]
            self.application.trainer.train_xgb(base_model_name=base_model_name,
                                                max_depth=max_depth, n_estimators=n_estimators)
        elif selected_type == "resnet_model_selected":
            resnet_layersizes=settings_dict["layersize"]
            resnet_blocks=settings_dict["num_blocks"]
            resnet_dropout=settings_dict["dropout"]
            max_epochs=settings_dict["max_epochs"]
            patience=settings_dict["patience"]
            batchsize=settings_dict["batchsize"]
            self.application.trainer.data_settings = self.application.trainer.data_settings._replace(batchsize=batchsize)
            self.application.trainer.train_resnet(base_model_name=base_model_name,
                                                    resnet_dropout=resnet_dropout,
                                                    resnet_layersizes=resnet_layersizes,
                                                    resnet_blocks=resnet_blocks,
                                                    max_epochs=max_epochs, patience=patience, colors=colors)
        elif selected_type == "mlp_model_selected":
            hidden_layers=settings_dict["hidden_layers"]
            mlp_dropout=settings_dict["dropout"]
            max_epochs=settings_dict["max_epochs"]
            patience=settings_dict["patience"]
            batchsize=settings_dict["batchsize"]
            self.application.trainer.data_settings = self.application.trainer.data_settings._replace(batchsize=batchsize)
            self.application.trainer.train_mlp(base_model_name=base_model_name, mlp_hidden_layers=hidden_layers,
                                                mlp_dropout=mlp_dropout, max_epochs=max_epochs, patience=patience, colors=colors)

    def compute_errors(self):
        """Compute errors for the selected dataset in the application."""
        trainer = self.application.trainer
        if trainer is None:
            print("Load data first!")
            return

        if len(trainer.models) == 0:
            print("Load models first!")
            return

        start = timer()
        trainer.compute_errors()
        print(f"compute errors took {(timer() - start):.2f} s.")

    def get_best_model(self):
        """Retrieve the best model based on the computed errors."""
        dataset_selected = self.application.interactive_plot.dataset_selection
        trainer = self.application.trainer

        if trainer is None:
            print("Load data first!")
            return

        if len(trainer.models) == 0:
            print("Load models first!")
            return

        if len(trainer.errors) == 0:
            print("Compute errors first!")
            return

        if len(trainer.errors[dataset_selected]) == 0:
            print("Compute errors first!")
            return

        last_best_model_path = self.best_model.path if self.best_model is not None else None
        start = timer()

        loss_selected = self.application.trainer.metric
        self._get_best_model(dataset_selection=dataset_selected, loss_selection=loss_selected)

        if self.best_model.path != last_best_model_path:
            self.best_model_was_changed = True
        else:
            self.best_model_was_changed = False

        print(f"get best model took {(timer() - start):.2f} s.")
