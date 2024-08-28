import json
import os
import ast
from fastai.tabular.all import *
from SAInT.common import set_seed, makedirs, load_json_dict
from SAInT.dataloader import create_dataloader
from SAInT.data_settings import load_data_settings_file
from SAInT.networks import FastAIModel, MLP, TabResNet, RandomForestModel, XGBModel, SVModel, MultiOutputRandomForestModel, do_grid_search
from SAInT.data_visualizer import DataVisualizer
from SAInT.modelloader import ModelLoader
from SAInT.metric import get_list_of_supported_standard_metrics, get_list_of_standard_metric_functions, get_metric
from SAInT.networks.fastai.stop_training_callback import StopTrainingCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Trainer():

    def __init__(self,
                 data_folder: str = None,
                 data_settings_path: str = None,
                 model_folder: str = None,
                 figure_folder: str = None,
                 random_seed: int = 123456):
        self.data_folder = data_folder
        self.data_settings_path = data_settings_path
        self.model_folder = model_folder
        self.figure_folder = figure_folder
        self.data_settings = None
        self.dataloader = None
        self.models = {}
        self.random_seed = random_seed
        self.errors = {}
        set_seed(random_seed)

    @staticmethod
    def load_or_train_model(model):
        model.load()
        if not model.is_trained():
            print(f"train {model.name}")
            model.fit()
            model.save()
        else:
            print(f"loaded {model.name}")
        return model

    @property
    def mode(self):
        if self.data_settings is None:
            return None
        return self.data_settings.mode

    @property
    def metric(self):
        if self.data_settings is None:
            return None
        return self.data_settings.metric

    @property
    def target_names(self):
        if self.dataloader is None:
            return None
        return list(self.dataloader.train.outputs.columns)

    @property
    def input_names(self):
        if self.dataloader is None:
            return None
        return list(self.dataloader.train.inputs.columns)

    @property
    def model_type_dict(self):
        output_size = len(self.target_names)
        model_type_dict = {"xgb": XGBModel, "rf": RandomForestModel}
        if output_size > 1:
            model_type_dict["mor_rf"] = MultiOutputRandomForestModel
        else:
            model_type_dict["sv"] = SVModel
        return model_type_dict

    @property
    def general_model_params(self):
        if self.dataloader is None or self.data_settings is None:
            return {}
        general_model_params = {
            "num_features": self.dataloader.train.inputs.shape[1],
            "output_size": len(self.target_names),
            "target_names": self.target_names,
            "categorical_cols": self.dataloader.train.categorical,
            "continuous_cols": self.dataloader.train.continuous,
            "mode": self.mode,
            "verbose": self.data_settings.verbose
        }
        return general_model_params

    def load_data(self,
                  dtype=None,
                  procs=None,
                  do_one_hot_encoding: bool = False,
                  valid_frac: float = 0.15,
                  test_frac: float = 0.15):
        if self.dataloader is not None:
            raise RuntimeError("Data is already loaded!")

        if self.data_settings_path is None:
            self.data_settings_path = f"{self.data_folder}/data_settings.json"

        if not os.path.exists(self.data_settings_path):
            raise RuntimeError(f"{self.data_settings_path} does not exist!")

        self.data_settings = load_data_settings_file(self.data_settings_path, self.data_folder)
        if self.data_settings is not None:
            if self.metric not in get_list_of_supported_standard_metrics() + [None]:
                raise RuntimeError(f"Unsupported metric {self.metric}!")
            if self.data_settings.output_names is None:
                raise RuntimeError(f"Select output first!")
            output_folder = self.data_settings.output_folder
            self.model_folder = f"{output_folder}/models/".replace("//", "/")
            makedirs(self.model_folder)
            self.figure_folder = f"{output_folder}/figures/".replace("//", "/")
            makedirs(self.figure_folder)
            batchsize = ast.literal_eval(str(self.data_settings.batchsize))
            self.data_settings = self.data_settings._replace(batchsize=batchsize)

        self.dataloader = create_dataloader(
            data_settings=self.data_settings,
            dtype=dtype,
            procs=procs,
            do_one_hot_encoding=do_one_hot_encoding,
            valid_frac=valid_frac,
            test_frac=test_frac)

        for batchsize in self.data_settings.batchsize:
            for mode, dataset in self.dataloader.datasets.items():
                if dataset is None:
                    print(f"no {mode} dataset")
                    continue
                if dataset.num_samples > 0 and dataset.num_samples < batchsize:
                    err_msg = f"Amount of available {mode} samples is smaller than batchsize of {batchsize}."
                    raise RuntimeError(err_msg)

    def load_models(self):
        modelloader = ModelLoader(
            model_folder=self.model_folder,
            dataloader=self.dataloader,
            random_seed=self.random_seed,
            general_model_params=self.general_model_params,
            models=self.models)
        models = modelloader.load_models()
        for model_name, model in models.items():
            self.models[model_name] = model
    def load_model(self, model_filename: str):
        modelloader = ModelLoader(
            model_folder=self.model_folder,
            dataloader=self.dataloader,
            random_seed=self.random_seed,
            general_model_params=self.general_model_params,
            models=self.models)
        models = modelloader.load_model(model_filename=model_filename)
        for model_name, model in models.items():
            self.models[model_name] = model

    def load_best_settings_from_file(self, best_params_path):
        return load_json_dict(best_params_path)

    def save_best_settings_to_file(self, best_params, best_params_path):
        json.dump(best_params, open(best_params_path, 'w'), indent=4, ensure_ascii=False)

    def grid_search(self, search_space_path, best_params_path, model_class, model_params):
        search_space = load_json_dict(search_space_path)
        if bool(search_space):
            clf = do_grid_search(model=model_class(**model_params),
                                    search_space=search_space)
            best_params = clf.best_params_
            self.save_best_settings_to_file(best_params, best_params_path)

    def create_classical_models(self,
                                base_model_name: str,
                                perform_new_grid_search: bool = True,
                                params_from_panel = None
                                ):
        model_folder = self.model_folder
        # Get general model parameters
        general_model_params = self.general_model_params

        # Classical models
        model_params = {
            "dls": self.dataloader.dls_train,
            "name": "tmp",
            "path": model_folder + "tmp.pkl",
            "random_seed": self.random_seed,
            **general_model_params
        }

        model_type_dict = self.model_type_dict
        for model_type_name, model_class in model_type_dict.items():
            print(f"Check if {model_type_name} should be trained ...")

            best_params = {}
            search_space_path = f"{self.data_folder}/{model_type_name}_search_space.json"
            best_params_path = f"{self.data_folder}/{model_type_name}_best_params.json"
            if perform_new_grid_search:
                print(f"Started grid search for {model_type_name} ...")
                self.grid_search(search_space_path=search_space_path,
                                 best_params_path=best_params_path,
                                 model_class=model_class, model_params=model_params)
            else:
                best_params = self.load_best_settings_from_file(best_params_path)

                if params_from_panel is not None:
                    print(f"Updated {model_type_name} params from panel ...")
                    if f"{model_type_name}_max_depth" in params_from_panel.keys():
                        best_params["max_depth"] = params_from_panel[f"{model_type_name}_max_depth"]
                    if f"{model_type_name}_n_estimators" in params_from_panel.keys():
                        best_params["n_estimators"] = params_from_panel[f"{model_type_name}_n_estimators"]
                else:
                    if bool(best_params):
                        print(f"Loaded best {model_type_name} params.")

            if bool(best_params):
                print("used params: ", best_params)
                params_str = "_".join(f"{param}_{value}"
                                      for param, value in best_params.items())
                model_name = f"{base_model_name}_{model_type_name}_{params_str}"
                model_filename = f"{model_folder}/{model_name}.pkl"
                params = {
                    "dls": self.dataloader.dls_train,
                    "name": model_name,
                    "path": model_filename,
                    "random_seed": self.random_seed,
                    **general_model_params,
                    **best_params
                }
                model_best = model_class(**params)
                self.models[model_name] = self.load_or_train_model(model_best)

    def create_fastai_models(self,
                             base_model_name: str,
                             mlp_hidden_layers: list = None,
                             mlp_dropout: list = None,
                             resnet_dropout: list = None,
                             resnet_layersizes: list = None,
                             resnet_blocks: list = None,
                             max_epochs: int = 500,
                             patience: int = None,
                             lr_max: float = None,
                             wd: float = None,
                             colors: list = ['orange', 'blue']
                             ):
        model_folder = self.model_folder
        num_features, output_size, lr_wd_str, fastaimodel_params, train_param_str = self._prepare_fastai_model_training(
            max_epochs=max_epochs, patience=patience, lr_max=lr_max, wd=wd)
        # Loop through different batch sizes
        batchsize_list = self.data_settings.batchsize
        print(f"batchsize = {batchsize_list}")
        for batchsize in batchsize_list:
            self.dataloader.batchsize = batchsize
            learner_params = self._get_learner_params(patience=patience, wd=wd, colors=colors)

            bs_lr_wd_str = f"bs_{batchsize}_{lr_wd_str}"

            # Create default model
            if self.mode == "regression" and batchsize > 1:
                default_tabular_learner = tabular_learner(**learner_params)
                default_tabular_learner_model_name = f"{base_model_name}_fastai_tabular_learner_{bs_lr_wd_str}_{train_param_str}"
                default_tabular_learner_model_filename = f"{model_folder}/models/{default_tabular_learner_model_name}.pth"
                default_tabular_learner_model = FastAIModel(
                    model=default_tabular_learner,
                    name=default_tabular_learner_model_name,
                    path=default_tabular_learner_model_filename,
                    **fastaimodel_params)
                self.models[
                    default_tabular_learner_model_name] = self.load_or_train_model(
                        default_tabular_learner_model)

            # Create MLP models
            for dropout in mlp_dropout:
                for hidden_layers in mlp_hidden_layers:
                    hidden_str = "_".join(
                        [str(hidden_layer) for hidden_layer in hidden_layers])
                    mlp_learner = TabularLearner(model=MLP(num_features,
                                                           output_size,
                                                           hidden_layers,
                                                           p=dropout),
                                                 **learner_params)
                    mlp_model_name = f"{base_model_name}_fastai_mlp_{bs_lr_wd_str}_hidden_{hidden_str}_p{dropout:.2f}_{train_param_str}"
                    mlp_model_filename = f"{model_folder}/models/{mlp_model_name}.pth"
                    mlp_model = FastAIModel(model=mlp_learner,
                                            name=mlp_model_name,
                                            path=mlp_model_filename,
                                            **fastaimodel_params)
                    self.models[mlp_model_name] = self.load_or_train_model(
                        mlp_model)

            # Create ResNet models
            # Check ResNet input parameters
            if len(resnet_layersizes) > 0 and len(resnet_blocks) == 0:
                raise RuntimeError(
                    "If ResNet shall be trained, resnet_blocks must be specified!")
            if len(resnet_blocks) > 0 and len(resnet_layersizes) == 0:
                raise RuntimeError(
                    "If ResNet shall be trained, resnet_layersizes must be specified!"
                )
            if batchsize > 1:
                for dropout in resnet_dropout:
                    for layersize in resnet_layersizes:
                        for blocks in resnet_blocks:
                            resnet_learner = TabularLearner(model=TabResNet(num_features, output_size, bn=False, p=dropout, size=layersize, num_blocks=blocks),
                                                            **learner_params)
                            resnet_model_name = f"{base_model_name}_fastai_res_{bs_lr_wd_str}_layersize_{layersize}_blocks_{blocks}_p{dropout:.2f}_{train_param_str}"
                            resnet_model_filename = f"{model_folder}/models/{resnet_model_name}.pth"
                            resnet_model = FastAIModel(
                                model=resnet_learner,
                                name=resnet_model_name,
                                path=resnet_model_filename,
                                **fastaimodel_params)
                            self.models[
                                resnet_model_name] = self.load_or_train_model(
                                    resnet_model)

    def create_models(self,
                      base_model_name: str,
                      mlp_hidden_layers: list = None,
                      mlp_dropout: list = None,
                      resnet_dropout: list = None,
                      resnet_layersizes: list = None,
                      resnet_blocks: list = None,
                      max_epochs: int = 500,
                      patience: int = None,
                      lr_max: float = None,
                      wd: float = None,
                      perform_new_grid_search: bool = True,
                      params_from_panel = None):
        self.create_fastai_models(base_model_name=base_model_name,
                                  mlp_hidden_layers=mlp_hidden_layers, mlp_dropout=mlp_dropout,
                                  resnet_dropout=resnet_dropout, resnet_layersizes=resnet_layersizes, resnet_blocks=resnet_blocks,
                                  max_epochs=max_epochs, patience=patience,
                                  lr_max=lr_max, wd=wd)
        self.create_classical_models(base_model_name=base_model_name, perform_new_grid_search=perform_new_grid_search,
                                     params_from_panel=params_from_panel)
        return self.models

    def _prepare_fastai_model_training(self, max_epochs: int = 500, patience: int = None, lr_max: float = None, wd: float = None):
        # Get general model parameters
        general_model_params = self.general_model_params
        output_size = general_model_params["output_size"]
        num_features = general_model_params["num_features"]
        lr_wd_str = self._get_lr_wd_str(lr_max=lr_max, wd=wd)
        # Prepare fastai model parameters
        fastaimodel_params = self._get_fastai_model_params(max_epochs=max_epochs, lr_max=lr_max)
        train_param_str = f"max_epochs_{max_epochs}_patience_{patience}"
        return num_features, output_size, lr_wd_str, fastaimodel_params, train_param_str

    def _get_cbs_list(self, patience: int = None, colors: list = ['orange', 'blue'], model_folder: str = None):

        def get_valid_values(recorder):
            row_index = recorder.metric_names.index('valid_loss')-1
            return [row[row_index] for row in recorder.values]

        def get_last_valid_value(recorder):
            row_index = recorder.metric_names.index('valid_loss')-1
            return recorder.values[-1:][0][row_index]

        def get_num_iterations_per_epochs(recorder) -> int:
            train_values = recorder.losses
            valid_values = get_valid_values(recorder)
            print("num train: ", len(train_values))
            print("num valid: ", len(valid_values))
            return int( len(train_values) / len(valid_values) )

        def get_valid_X(recorder) -> list:
            step = get_num_iterations_per_epochs(recorder)
            return list(range(step, len(recorder.losses)+step, step))

        def plot_learning_curve(recorder, colors):
            plt.clf()
            #recorder.plot_loss()
            # Create custom plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(recorder.losses)),
                     recorder.losses, label='Training Loss', color=colors[0])
            if len(recorder.losses) > 0:
                plt.plot(
                    get_valid_X(recorder),
                    [row[recorder.metric_names.index('valid_loss')-1] for row in recorder.values],
                    label='Validation Loss', color=colors[1])
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss: final valid_loss = {get_last_valid_value(recorder)}')
            plt.legend()
            plt.grid(True)

        cbs = []
        if self.dataloader.valid is not None:
            cbs.append(
                ShowGraphCallback(after_fit=lambda self: (
                    plot_learning_curve(self.learn.recorder, colors),
                    #plt.ylim(0, 1.0),
                    plt.savefig(f"{model_folder}/models/learning_curve.svg",
                                facecolor='white'),
                    plt.clf()
                    )), )
            cbs.append(
                SaveModelCallback(monitor="valid_loss",
                                  comp=np.less,
                                  min_delta=0.0,
                                  fname="model",
                                  every_epoch=False,
                                  at_end=False,
                                  with_opt=False,
                                  reset_on_fit=True))
            if patience is not None:
                cbs.append(
                    EarlyStoppingCallback(monitor="valid_loss",
                                          comp=np.less,
                                          min_delta=0.0,
                                          patience=patience,
                                          reset_on_fit=True))
            cbs.append(StopTrainingCallback())
        return cbs

    def _get_fastai_model_params(self, max_epochs: int = 500, lr_max: float = None):
        fastaimodel_params = {
            "max_epochs": max_epochs,
            "lr_max": lr_max,
            **self.general_model_params
        }
        return fastaimodel_params

    def _get_learner_params(self, patience: int = None, wd: float = None, colors: list = ['orange', 'blue']):
        # Prepare common callback list
        cbs = self._get_cbs_list(patience=patience, colors=colors, model_folder=self.model_folder)
        loss_func = get_metric(self.metric)
        # metrics = get_list_of_standard_metric_functions()
        metrics = [mae, mse]
        learner_params = {
            "dls": self.dataloader.dls_train,
            "cbs": cbs,
            "metrics": metrics,
            "loss_func": loss_func,
            "opt_func": Adam,
            "wd": wd,
            "path": self.model_folder
        }
        return learner_params

    def _get_lr_wd_str(self, lr_max: float = None, wd: float = None):
        lr_str = str(lr_max) if lr_max is not None else "default"
        wd_str = str(wd) if wd is not None else "default"
        return f"lr_{lr_str}_wd_{wd_str}"

    def train_mlp(self, base_model_name: str, mlp_hidden_layers: list = None, mlp_dropout: list = None,
                  max_epochs: int = 500, patience: int = None, lr_max: float = None, wd: float = None, colors: list = ['orange', 'blue']):
        model_folder = self.model_folder
        num_features, output_size, lr_wd_str, fastaimodel_params, train_param_str = self._prepare_fastai_model_training(
            max_epochs=max_epochs, patience=patience, lr_max=lr_max, wd=wd)

        # Loop through different batch sizes
        batchsize_list = self.data_settings.batchsize
        print(f"batchsize = {batchsize_list}")
        for batchsize in batchsize_list:
            self.dataloader.batchsize = batchsize

            learner_params = self._get_learner_params(patience=patience, wd=wd, colors=colors)

            bs_lr_wd_str = f"bs_{batchsize}_{lr_wd_str}_{self.metric}"

            for dropout in mlp_dropout:
                for hidden_layers in mlp_hidden_layers:
                    hidden_str = "_".join(
                        [str(hidden_layer) for hidden_layer in hidden_layers])
                    mlp_learner = TabularLearner(model=MLP(num_features,
                                                            output_size,
                                                            hidden_layers,
                                                            p=dropout),
                                                    **learner_params)
                    mlp_model_name = f"{base_model_name}_fastai_mlp_{bs_lr_wd_str}_hidden_{hidden_str}_p{dropout:.2f}_{train_param_str}"
                    mlp_model_filename = f"{model_folder}/models/{mlp_model_name}.pth"
                    mlp_model = FastAIModel(model=mlp_learner,
                                            name=mlp_model_name,
                                            path=mlp_model_filename,
                                            **fastaimodel_params)
                    self.models[mlp_model_name] = self.load_or_train_model(
                        mlp_model)

    def train_resnet(self, base_model_name: str, resnet_dropout: list = None, resnet_layersizes: list = None, resnet_blocks: list = None,
                     max_epochs: int = 500, patience: int = None, lr_max: float = None, wd: float = None, colors: list = ['orange', 'blue']):
        model_folder = self.model_folder
        print(f"train_resnet -> model_folder: {model_folder}")
        num_features, output_size, lr_wd_str, fastaimodel_params, train_param_str = self._prepare_fastai_model_training(
            max_epochs=max_epochs, patience=patience, lr_max=lr_max, wd=wd)

        # Loop through different batch sizes
        batchsize_list = self.data_settings.batchsize
        print(f"batchsize = {batchsize_list}")
        for batchsize in batchsize_list:
            self.dataloader.batchsize = batchsize

            learner_params = self._get_learner_params(patience=patience, wd=wd, colors=colors)

            bs_lr_wd_str = f"bs_{batchsize}_{lr_wd_str}_{self.metric}"

            if batchsize > 1:
                for dropout in resnet_dropout:
                    for layersize in resnet_layersizes:
                        for blocks in resnet_blocks:
                            resnet_learner = TabularLearner(model=TabResNet(num_features, output_size, bn=False, p=dropout, size=layersize, num_blocks=blocks),
                                                            **learner_params)
                            resnet_model_name = f"{base_model_name}_fastai_res_{bs_lr_wd_str}_layersize_{layersize}_blocks_{blocks}_p{dropout:.2f}_{train_param_str}"
                            resnet_model_filename = f"{model_folder}/models/{resnet_model_name}.pth"
                            resnet_model = FastAIModel(
                                model=resnet_learner,
                                name=resnet_model_name,
                                path=resnet_model_filename,
                                **fastaimodel_params)
                            self.models[
                                resnet_model_name] = self.load_or_train_model(
                                    resnet_model)

    def _train_single_classical(self, base_model_name, model_type_name, model_class, best_params):
        model_folder = self.model_folder
        # Get general model parameters
        general_model_params = self.general_model_params
        params_str = "_".join(f"{param}_{value}"
                                for param, value in best_params.items())
        model_name = f"{base_model_name}_{model_type_name}_{params_str}"
        model_filename = f"{model_folder}/{model_name}.pkl"
        params = {
            "dls": self.dataloader.dls_train,
            "name": model_name,
            "path": model_filename,
            "random_seed": self.random_seed,
            **general_model_params,
            **best_params
        }
        model_best = model_class(**params)
        self.models[model_name] = self.load_or_train_model(model_best)

    def train_rf(self, base_model_name: str, max_depth: int = None, n_estimators: int = None):
        best_params = {"max_depth": max_depth, "n_estimators": n_estimators}
        self._train_single_classical(base_model_name=base_model_name,
                                    model_type_name = "rf",
                                    model_class=RandomForestModel,
                                    best_params=best_params)

    def train_xgb(self, base_model_name: str, max_depth: int = None, n_estimators: int = None):
        best_params = {"max_depth": max_depth, "n_estimators": n_estimators}
        self._train_single_classical(base_model_name=base_model_name,
                                    model_type_name = "xgb",
                                    model_class=XGBModel,
                                    best_params=best_params)


    def create_histograms(self, do_save: bool = True, do_show: bool = True):
        self.dataloader.create_histograms(figure_folder=self.figure_folder, do_save=do_save, do_show=do_show)


    def compute_errors(self):
        def compute_error_on_model(model, data, metric):
            error, _ = model.test(data, metric=metric)
            return error

        datasets = {
            "train": self.dataloader.dls_train,
            "valid": self.dataloader.to_valid,
            "test": self.dataloader.to_test
        }
        for mode, data in datasets.items():
            if mode not in self.errors.keys():
                self.errors[mode] = {}
            if self.metric not in self.errors[mode].keys():
                self.errors[mode][self.metric] = {}
            for model in self.models.values():
                if model not in self.errors[mode][self.metric].keys():
                    self.errors[mode][self.metric][model.name] = compute_error_on_model(model, data, self.metric)
                error = self.errors[mode][self.metric][model.name]
                print(f"{model.name}:\n   {mode} data: {self.metric}: {error:.5f}")

    def plot_model_pred_plots(self,
                              figsize: tuple = None,
                              ds_for_model_selection: str = "valid") -> dict:

        def get_model_type(model_name: str) -> str:
            for k in self.model_type_dict.keys():
                if k in model_name:
                    return k
            if "_mlp_" in model_name:
                return "mlp"
            if "_res_" in model_name:
                return "resnet"

        if self.mode != "regression":
            raise RuntimeError("Only regression mode supported.")

        preds = {}
        if ds_for_model_selection == "valid":
            ds = self.dataloader.valid
        elif ds_for_model_selection == "test":
            ds = self.dataloader.test
        else:
            raise RuntimeError(
                f"Only 'valid' and 'test' supported, got '{ds_for_model_selection}'."
            )

        output_name = self.dataloader.train.output_names[0]

        X = ds.inputs
        y = ds.outputs[output_name]

        for model_name, model in self.models.items():
            preds[get_model_type(model_name)] = model.predict(X).flatten()

        DataVisualizer.plot_pred_err_display(y_test=y,
                              y_pred_all=preds,
                              figsize=figsize,
                              filepath=self.figure_folder + "preds",
                              do_save=True)

    def get_best_model(self, error_dict: dict):
        if not error_dict:
            print("Can't get best model, no error data available!")
            return None
        sorted_models = sorted(self.models.items(),
                               key=lambda x: error_dict[x[0]])
        print(
            f"Models sorted by error: {[model[0] for model in sorted_models]}\n"
        )

        best_model_name, best_model = sorted_models[0]
        print(
            f"Best model: {best_model_name}: error = {error_dict[best_model_name]:.5f}"
        )

        return best_model
