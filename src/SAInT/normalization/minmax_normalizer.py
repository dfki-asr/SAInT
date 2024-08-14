from collections import namedtuple
from typing import List
from SAInT.data_normalizer import Normalizer

NormalizationMinMaxValues = namedtuple('NormalizationMinMaxValues', 'min_values, max_values')

class MinMaxNormalizer(Normalizer):
    def __init__(self, normalization_values: NormalizationMinMaxValues = None,
                 features_to_normalize: List[str] = None, verbose: bool = False):
        super().__init__(normalization_values=normalization_values,
                 features_to_normalize=features_to_normalize, verbose=verbose)

    def _compute_normalization_values(self, dataframe):
        self.normalization_values = NormalizationMinMaxValues(
                min_values=dataframe.min(skipna=True),
                max_values=dataframe.max(skipna=True))

    def _perform_normalization(self, dataframe, selected_features):
        column_min = self.normalization_values.min_values
        column_max = self.normalization_values.max_values
        selected_column_min = column_min[selected_features]
        selected_column_max = column_max[selected_features]
        # Check if lengths of selected features, min and max match
        if len(selected_features) != len(selected_column_min) or len(selected_features) != len(selected_column_max):
            raise ValueError("Lengths of selected features, min and max do not match.")
        dataframe.loc[:, selected_features] = \
            (dataframe.loc[:, selected_features] - selected_column_min) / (selected_column_max - selected_column_min)
        return dataframe

    def _perform_denormalization(self, dataframe, selected_features):
        column_min = self.normalization_values.min_values
        column_max = self.normalization_values.max_values
        selected_column_min = column_min[selected_features]
        selected_column_max = column_max[selected_features]
        # Check if lengths of selected features, min and max match
        if len(selected_features) != len(selected_column_min) or len(selected_features) != len(selected_column_max):
            raise ValueError("Lengths of selected features, min and max do not match.")
        dataframe.loc[:, selected_features] = \
            dataframe.loc[:, selected_features] * (selected_column_max - selected_column_min) + selected_column_min
        return dataframe
