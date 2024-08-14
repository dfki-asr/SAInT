import pickle
from typing import List, Dict
import numpy as np
import pandas as pd
from SAInT.common import makedirs


class Normalizer:
    def __init__(self, normalization_values = None,
                 features_to_normalize: List[str] = None, verbose: bool = False):
        self.normalization_values = normalization_values
        self.features_to_normalize = features_to_normalize or []
        self.verbose = verbose
        self.is_normalized = False

    def _compute_normalization_values(self, dataframe):
        pass

    def _perform_normalization(self, dataframe, selected_features):
        pass

    def _perform_denormalization(self, dataframe, selected_features):
        pass

    def normalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Normalize the DataFrame."""
        if dataframe is None:
            raise RuntimeError("Dataframe is None!")
        if self.normalization_values is None:
            if self.verbose:
                print(f"Calculate normalization values for {self.mode}.")
            self._compute_normalization_values(dataframe)
        if self.features_to_normalize is None:
            raise RuntimeError("No features selected to be normalized!")
        if len(self.features_to_normalize) == 0:
            raise RuntimeError("No features selected to be normalized!")
        if self.verbose:
            print(f"Normalizing Features: {self.features_to_normalize}")
        selected_features = self.features_to_normalize
        # Check if selected features exist in the DataFrame
        if not all(feature in dataframe.columns for feature in selected_features):
            raise ValueError("Selected features not found in DataFrame columns.")
        # Normalize the selected columns in the copied DataFrame
        dataframe = self._perform_normalization(dataframe, selected_features)
        if self.verbose:
            print(f"Normalization successful.")
        self.is_normalized = True
        return dataframe

    def denormalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Denormalize the DataFrame if it has been normalized."""
        if dataframe is None:
            raise RuntimeError("Dataframe is None!")
        if not self.is_normalized:
            print("No denormalization necessary (data is not normalized)")
            return dataframe
        if self.normalization_values is None:
            raise RuntimeError("Normalization values are not set!")
        selected_features = self.features_to_normalize
        dataframe = self._perform_denormalization(dataframe, selected_features)
        if self.verbose:
            print(f"Denormalization successful.")
        self.is_normalized = False
        return dataframe

    def set_normalization_values(self, normalization_values: Dict[str, float]) -> None:
        """Set normalization values for the dataset."""
        if self.verbose:
            print("Set normalization values.")
        self.normalization_values = normalization_values

    def _dump_to_file(self, output_folder, name, values):
        pickle.dump(values, open(output_folder + f'/{name}.pkl', 'wb'))

    def save_normalization_values(self, output_folder: str) -> None:
        """Save normalization values to the specified output folder."""
        if self.verbose:
            print(f"Save normalization values to folder {output_folder}.")
        makedirs(output_folder)
        for name in self.normalization_values._fields:
            values = getattr(self.normalization_values, name)
            if values is not None:
                self._dump_to_file(output_folder, name, values)
