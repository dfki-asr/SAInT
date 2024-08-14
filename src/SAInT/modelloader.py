import os
import joblib
from fastai.learner import load_learner
from pathlib import Path
from SAInT.networks import FastAIModel, RandomForestModel, XGBModel, SVModel, MultiOutputRandomForestModel


def list_pkl_files(folder_path: str):
    folder = Path(folder_path)
    return [f.name for f in folder.glob("*.pkl") if f.is_file()]


def load_model_from_joblib(model_filename: str):
    return joblib.load(model_filename)


def load_model_from_fastai(model_filename: str, data_loaders):
    learner = load_learner(model_filename)
    path = Path(os.path.abspath(os.path.dirname(model_filename))).parent
    learner.path = path
    learner.dls = data_loaders
    return learner


class ModelLoader:

    def __init__(self, model_folder, dataloader, random_seed,
                 general_model_params, models=None):
        self.model_folder = model_folder
        self.dataloader = dataloader
        self.random_seed = random_seed
        self.general_model_params = general_model_params
        self.models = models if models is not None else {}


    @property
    def target_names(self):
        return list(
            self.dataloader.train.outputs.columns) if self.dataloader else None

    def insert_model(self, model_name: str, model_filename: str, data_loaders):
        model_type_dict = {
            "xgb": XGBModel,
            "rf": RandomForestModel,
            "fastai": FastAIModel,
            "mor_rf": MultiOutputRandomForestModel,
            "sv": SVModel
        }
        model_type = next(
            (t for t in model_type_dict.keys() if f"{t}_" in model_name),
            None)
        if model_type:
            model = load_model_from_joblib(
                model_filename
            ) if model_type != "fastai" else load_model_from_fastai(
                model_filename, data_loaders)
            model_class = model_type_dict[model_type]
            model_filename = model_filename.replace(
                ".pkl",
                ".pth") if model_type == "fastai" else model_filename
            model_params = {
                "name": model_name,
                "path": model_filename,
                "model": model,
                **self.general_model_params
            }
            if model_type != "fastai":
                model_params["dls"] = data_loaders
                model_params["random_seed"] = self.random_seed
            else:
                model_params["max_epochs"] = 1

            model = model_class(**model_params)
            if model_type == "fastai":
                model.load()
            self.models[model_name] = model

    def insert_models(self, folder_path: str, data_loaders):
        for filename in list_pkl_files(folder_path):
            model_name, ext = os.path.splitext(filename)
            if ext != ".pkl" or model_name in {"mean_values", "std_values"}:
                continue
            model_filename = f"{folder_path}/{model_name}.pkl"
            self.insert_model(model_name, model_filename, data_loaders)

    def load_models(self):
        self.insert_models(self.model_folder,
                           self.dataloader.dls_train)
        self.insert_models(f"{self.model_folder}/models/",
                           self.dataloader.dls_train)
        return self.models

    def load_model(self, model_filename: str):
        model_basename = os.path.basename(model_filename)
        model_name, ext = os.path.splitext(model_basename)
        self.insert_model(model_name,
                          model_filename,
                          self.dataloader.dls_train)
        return self.models