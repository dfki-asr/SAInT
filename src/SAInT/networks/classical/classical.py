import abc
import numpy as np
import pandas as pd
from typing import Tuple, Union
import joblib
from fastai.tabular.all import TabularDataLoaders
from SAInT.model import Model
from SAInT.common import exists
import SAInT.metrics as metrics


class ClassicalModel(Model):

    def __init__(self,
                 model,
                 name: str,
                 path: str,
                 num_features: int,
                 output_size: int,
                 target_names: list,
                 categorical_cols: list,
                 continuous_cols: list,
                 mode: str,
                 sample_weight: list = None,
                 verbose: bool = False):
        self.model = None
        self.weights_loaded = False
        self.inputs = None
        self.outputs = None
        self.feature_importances = None
        self.sample_weight = sample_weight
        super().__init__(model, name, path, num_features, output_size,
                         target_names, categorical_cols, continuous_cols, mode,
                         verbose)

    def load(self) -> None:
        if exists(self.path):
            self.model = joblib.load(self.path)
            self.weights_loaded = True

    def save(self) -> None:
        if not exists(self.path):
            joblib.dump(self.model, self.path)

    def fit(self) -> None:
        self.model.fit(self.inputs,
                       self.outputs,
                       sample_weight=self.sample_weight)
        self.weights_loaded = True

    def get_output(self, sample: np.ndarray):
        return self.model.predict(sample)

    def predict_proba(self, sample: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.model.predict_proba(sample)

    def reshape_to_2D(self, arr):
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def test_input_and_output(self, inputs: np.ndarray, outputs: np.ndarray,
                              metric) -> Tuple[float, list]:
        pred_org = self.model.predict(inputs)

        outputs = self.reshape_to_2D(outputs)
        pred = self.reshape_to_2D(pred_org.copy())
        try:
            error_function = getattr(metrics, metric)
            error = error_function(outputs, pred)
        except AttributeError:
            raise RuntimeError(f"Unknown metric {metric}!")

        return error, pred_org

    def fast_evaluation_fct(self, sample: np.ndarray) -> np.ndarray:
        prediction = self.get_output(sample)
        if len(prediction.shape) == 1:
            return prediction.flatten()
        if prediction.shape[1] == 1:
            return prediction.flatten()
        return prediction

    @abc.abstractmethod
    def test(self, dls: TabularDataLoaders, metric) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def show_architecture(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_n_most_important_features(self, num_top_features: int) -> list:
        raise NotImplementedError
