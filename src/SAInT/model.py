from typing import Union
import numpy as np
import pandas as pd
from fastai.tabular.all import Module, TabularPandas, TabularLearner
from SAInT.common import makedirs
from SAInT.data_visualizer import DataVisualizer


class Model():

    def __init__(self,
                 model: Union[TabularLearner, Module],
                 name: str,
                 path: str,
                 num_features: int,
                 output_size: int,
                 target_names: list,
                 categorical_cols: list,
                 continuous_cols: list,
                 mode: str,
                 verbose: bool = False):
        super().__init__()
        self.model = model
        self.name = name
        self.path = path
        self.input_size = num_features
        self.output_size = output_size
        self.target_names = target_names
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.mode = mode
        self.verbose = verbose
        self.weights_loaded = False

    @property
    def input_names(self) -> list:
        return self.categorical_cols + self.continuous_cols

    def is_trained(self) -> bool:
        return self.weights_loaded

    def load(self):
        pass

    def save(self):
        pass

    def export_as_pkl(self):
        pass

    def fit(self):
        pass

    def get_output(self):
        pass

    def predict(self, sample: Union[pd.DataFrame,
                                           np.ndarray]) -> np.ndarray:
        if isinstance(sample, list):
            sample = np.ndarray(sample)
        if isinstance(sample, np.ndarray):
            if sample.ndim == 1:
                sample = sample.reshape(-1, self.input_size)
        output = self.get_output(sample)
        if isinstance(output, np.ndarray):
            if output.ndim == 1:
                output = output.reshape(-1, self.output_size)
        return output

    def predict_proba(self, sample: Union[pd.DataFrame,
                                           np.ndarray]) -> np.ndarray:
        pass

    def fast_evaluation_fct(self, sample: np.ndarray) -> np.ndarray:
        pass

    def test(self, dls, metric) -> float:
        pass

    def show_architecture(self):
        pass

    def get_prediction_groundtruth_errors(self, data: list):
        num_samples = len(data)
        groundtruth, predictions = [], []
        for idx in range(num_samples):
            sample = data.xs.iloc[idx:idx + 1]

            prediction = self.predict(sample).flatten()
            for i, value in enumerate(prediction):
                if i >= len(predictions):
                    predictions.append([])
                predictions[i].append(value)

            groundtruth_value = data.ys.iloc[idx:idx + 1].values.flatten()
            for i, value in enumerate(groundtruth_value):
                if i >= len(groundtruth):
                    groundtruth.append([])
                groundtruth[i].append(value)

        errors = [
            list(np.array(predictions[i]) - np.array(groundtruth[i]))
            for i in range(len(self.target_names))
        ]

        return predictions, groundtruth, errors

    def plot_prediction_groundtruth_errors(self,
                                           predictions: list,
                                           groundtruth: list,
                                           errors: list,
                                           metric: str,
                                           figure_folder: str = "",
                                           do_save: bool = True,
                                           do_show: bool = True,
                                           as_pdf: bool = True):
        makedirs(figure_folder)
        for i, target_name in enumerate(self.target_names):
            title = figure_folder + self.name + "_" + target_name
            joined_df = pd.DataFrame({
                "prediction": predictions[i],
                "groundtruth": groundtruth[i]
            })
            markersize = 1.0 if len(predictions[i]) > 100 else 5.0
            DataVisualizer.plot_values(
                data={'Prediction vs Groundtruth': joined_df},
                marked_samples=None,
                figsize=(16, 16),
                filepath=title + "_pred_gt",
                markersize=markersize,
                do_save=do_save,
                do_show=do_show
            )
            error_df = pd.DataFrame({f"error ({metric})": errors[i]})
            DataVisualizer.plot_histogram(
                error_df,
                figsize=(8, 8),
                filepath=title + "_error",
                do_save=do_save,
                do_show=do_show,
                as_pdf=as_pdf
            )

    def get_statistics(self,
                       data: TabularPandas,
                       metric: str,
                       figure_folder: str = "",
                       do_save: bool = True,
                       do_show: bool = True,
                       as_pdf: bool = True) -> dict:

        def _get_error_statistics(data: list) -> dict:
            # minimum and maximum
            min_idx, max_idx = np.argmin(data), np.argmax(data)
            min_value, max_value = data[min_idx], data[max_idx]

            # median
            sorted_data = np.sort(data)
            median_value = sorted_data[int((len(data) + 1.0) / 2.0)]
            median_idx = data.index(median_value)

            # best (lowest absolute value)
            abs_data = sorted(data, key=abs)
            best_value = abs_data[0]
            best_idx = data.index(best_value)

            statistics = {
                "min": (min_idx, min_value),
                "max": (max_idx, max_value),
                "median": (median_idx, median_value),
                "best": (best_idx, best_value)
            }
            return statistics

        predictions, groundtruth, errors = self.get_prediction_groundtruth_errors(
            data)
        if self.verbose is True or do_save is True:
            if do_save is True and figure_folder == '':
                pass
            else:
                self.plot_prediction_groundtruth_errors(
                    predictions=predictions,
                    groundtruth=groundtruth,
                    errors=errors,
                    metric=metric,
                    figure_folder=figure_folder,
                    do_save=do_save,
                    do_show=do_show,
                    as_pdf=as_pdf)

        statistics = {}
        for i, target_name in enumerate(self.target_names):
            statistics[target_name] = _get_error_statistics(errors[i])

        if self.verbose is True:
            print("\nerror statistics: \n", statistics)

        return statistics
