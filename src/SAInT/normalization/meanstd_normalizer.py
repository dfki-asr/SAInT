from collections import namedtuple
from typing import List
from SAInT.data_normalizer import Normalizer

NormalizationMeanStdValues = namedtuple('NormalizationMeanStdValues', 'mean_values, std_values')

class MeanStdNormalizer(Normalizer):
    def __init__(self, normalization_values: NormalizationMeanStdValues = None,
                 features_to_normalize: List[str] = None, verbose: bool = False):
        super().__init__(normalization_values=normalization_values,
                 features_to_normalize=features_to_normalize, verbose=verbose)

    def _compute_normalization_values(self, dataframe):
        self.normalization_values = NormalizationMeanStdValues(
                mean_values=dataframe.mean(skipna=True),
                std_values=dataframe.std(skipna=True))

    def _perform_normalization(self, dataframe, selected_features):
        column_mean = self.normalization_values.mean_values
        column_std = self.normalization_values.std_values
        selected_column_mean = column_mean[selected_features]
        selected_column_std = column_std[selected_features]
        # Check if lengths of selected features, mean and std match
        if len(selected_features) != len(selected_column_mean) or len(selected_features) != len(selected_column_std):
            raise ValueError("Lengths of selected features, mean and std do not match.")
        dataframe.loc[:, selected_features] = \
            (dataframe.loc[:, selected_features] - selected_column_mean) / selected_column_std
        return dataframe

    def _perform_denormalization(self, dataframe, selected_features):
        column_mean = self.normalization_values.mean_values
        column_std = self.normalization_values.std_values
        selected_column_mean = column_mean[selected_features]
        selected_column_std = column_std[selected_features]
        # Check if lengths of selected features, mean and std match
        if len(selected_features) != len(selected_column_mean) or len(selected_features) != len(selected_column_std):
            raise ValueError("Lengths of selected features, mean and std do not match.")
        dataframe.loc[:, selected_features] = \
            (dataframe.loc[:, selected_features] * selected_column_std) + selected_column_mean
        return dataframe
