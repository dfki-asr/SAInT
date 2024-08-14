import os
import numpy as np
import pandas as pd
from typing import Tuple, Union
from shutil import move as move_file
from fastai.tabular.all import torch, TabularDataLoaders, TabularLearner, \
    ShowGraphCallback, EarlyStoppingCallback, SaveModelCallback, F
from torch.nn.functional import softmax
from SAInT.model import Model
from SAInT.common import exists
import SAInT.metrics as metrics
from SAInT.networks.fastai.stop_training_callback import StopTrainingCallback


class FastAIModel(Model):

    def __init__(self,
                 model: TabularLearner,
                 name: str,
                 path: str,
                 num_features: int,
                 output_size: int,
                 target_names: list,
                 categorical_cols: list,
                 continuous_cols: list,
                 mode: str,
                 max_epochs: int = 350,
                 lr_max: float = None,
                 verbose: bool = False):
        self.model = None
        self.weights_loaded = False
        super().__init__(model, name, path, num_features, output_size,
                         target_names, categorical_cols, continuous_cols, mode,
                         verbose)
        self.max_epochs = max_epochs
        self.lr_max = lr_max

    def load(self) -> None:
        if exists(self.path):
            self.model.load(self.name, with_opt=True)
            self.model.freeze()
            self.model.eval()
            print(self.model.summary())
            self.weights_loaded = True

    def save(self) -> None:
        if not exists(self.path):
            self.model.save(self.name, with_opt=True)
            self.export_as_pkl()

    def export_as_pkl(self) -> None:
        self.model.remove_cbs(
            [ShowGraphCallback, EarlyStoppingCallback, SaveModelCallback, StopTrainingCallback])
        self.model.export("models/" + self.name + ".pkl")

    def fit(self) -> None:
        self.model.unfreeze()
        self.model.train()
        # Cut first epoch from learning curve plot
        self.model.fit_one_cycle(1)
        if self.max_epochs > 1:
            if self.lr_max is not None:
                self.model.fit_one_cycle(self.max_epochs-1, lr_max=self.lr_max)
            else:
                self.model.fit_one_cycle(self.max_epochs-1)
        model_folder = os.path.dirname(self.path)
        src_model_file = f"{model_folder}/model.pth"
        dst_model_file = f"{model_folder}/{self.name}.pth"
        move_file(src_model_file, dst_model_file)
        src_learning_curve_file = f"{model_folder}/learning_curve.svg"
        dst_learning_curve_file = f"{model_folder}/{self.name}_learning_curve.svg"
        if exists(src_learning_curve_file):
            move_file(src_learning_curve_file, dst_learning_curve_file)
        self.model.freeze()
        self.model.eval()
        print(self.model.summary())
        self.export_as_pkl()
        self.weights_loaded = True

    def forward(self, sample: np.ndarray) -> torch.tensor:
        sample = torch.tensor(
            np.array(sample, dtype=np.float32).reshape((-1, self.input_size)))
        axis = 1
        categorical, continuous = torch.split(
            sample, [len(self.categorical_cols),
                     len(self.continuous_cols)], axis)
        output = self.model.model.forward(categorical, continuous)
        return output

    def get_output(self, sample: np.ndarray):
        return self.forward(sample).detach().numpy()

    def predict_proba(self, sample: torch.Tensor) -> np.ndarray:
        output = self.forward(np.array(sample))
        if self.mode == 'classification':
            if self.output_size == 1:  # Binary classification
                output = torch.sigmoid(output)
            else:  # Multi-class classification
                output = F.log_softmax(output, dim=1)

            if self.output_size == 1:  # Binary classification
                probs = torch.cat([1 - output, output], dim=1)
                # Get probabilities for both classes
            else:  # Multi-class classification
                probs = torch.exp(output)
                # Since output is log_softmax, take exp to get probabilities
        else:
            raise ValueError("predict_proba is not applicable for regression tasks.")
        return probs.detach().numpy()

    def fast_evaluation_fct(self, sample: np.ndarray) -> np.ndarray:
        sample_reshaped = [v.reshape(1, self.input_size) for v in sample]
        output = self.get_output(sample_reshaped)
        if len(output.shape) == 1:
            return output.flatten()
        if output.shape[1] == 1:
            return output.flatten()
        return output

    def reshape_to_2D(self, arr):
        if len(arr.shape) == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def test(self, dls: TabularDataLoaders, metric: str) -> Tuple[float, list]:
        inputs = dls.xs.values
        outputs = dls.ys.values
        pred_org = []
        for sample in inputs:
            pred_org.extend(self.predict(sample))

        pred = np.array(pred_org.copy())
        outputs = self.reshape_to_2D(outputs)
        pred = self.reshape_to_2D(pred)
        try:
            error_function = getattr(metrics, metric)
            error = error_function(outputs, pred)
        except AttributeError:
            raise RuntimeError(f"Unknown metric {metric}!")
        return error, pred_org

    def show_architecture(self) -> None:
        print(self.model.summary())

    def show_results(self) -> None:
        self.model.show_results()
