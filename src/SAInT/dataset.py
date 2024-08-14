import pickle
from typing import Union, List, Dict
from itertools import chain
import numpy as np
import pandas as pd
from fastai.tabular.all import TabularPandas, TabularDataLoaders
from SAInT.normalization import MeanStdNormalizer, MinMaxNormalizer
from SAInT.data_balancer import DataBalancer

def custom_cont_cat_split(df: pd.DataFrame, max_card: int = 10) -> (List[str], List[str]):
    """
    Split DataFrame columns into continuous and categorical based on data type and cardinality.

    :param df (pd.DataFrame): Input DataFrame.
    :param max_card (int): Maximum cardinality for categorical features (default is 10).
    :return: tuple: List of continuous and categorical column names.
    """
    cont_names, cat_names = [], []
    for label in df.columns:
        dtype_str = str(df[label].dtype)
        num_unique_values = df[label].nunique()
        if "int" in dtype_str or "float" in dtype_str:
            cont_names.append(label)
        elif "object" in dtype_str and num_unique_values <= max_card:
            cat_names.append(label)
        else:
            cont_names.append(label)
    return cont_names, cat_names


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, mode: str = "", output_names: List[str] = None,
                 normalization_values = None, features_to_normalize: List[str] = None, normalization: str = "none",
                 verbose: bool = False, random_seed: int = 123456, is_normalized: bool = False,
                 categorical_names: List[str] = None, continuous_names: List[str] = None, procs: List = None):
        """
        Initialize the Dataset object.

        :param dataframe (pd.DataFrame): Input DataFrame.
        :param mode (str): Mode of the dataset (e.g., 'train', 'valid').
        :param output_names (list): List of output feature names.
        :param normalization_values: Normalization values object.
        :param features_to_normalize (list): List of features to normalize.
        :param normalization (str): Method: none, mean_std or min_max
        :param verbose (bool): Verbosity flag.
        :param random_seed (int): Random seed for reproducibility.
        :param is_normalized (bool): Flag indicating if the data is normalized.
        :param categorical_names (list): List of categorical feature names.
        :param continuous_names (list): List of continuous feature names.
        :param procs (list): List of preprocessing functions.
        """
        if dataframe is None:
            raise ValueError("Dataframe is None!")

        self.dataframe = dataframe
        self.mode = mode
        self.output_names = output_names or []
        self.verbose = verbose
        self.random_seed = random_seed
        self.categorical_names = categorical_names
        self.continuous_names = continuous_names
        self.procs = procs or []
        self.is_normalized = is_normalized
        self.dataframe = self.convert_object_or_bool_dtype(dataframe)
        self.normalization = normalization
        if normalization == "mean_std":
            self.normalizer = MeanStdNormalizer(normalization_values, features_to_normalize, verbose)
        else:
            self.normalizer = MinMaxNormalizer(normalization_values, features_to_normalize, verbose)
        self.balancer = DataBalancer()

        for f in self.output_names:
            if f not in dataframe.columns:
                raise ValueError(f"Output feature {f} is not part of {dataframe.columns}")

        self._check_for_nan_values()

    def convert_to_float(self, dtype=np.float64) -> pd.DataFrame:
        """Convert all columns in the DataFrame to a specified float type."""
        dataframe = self.dataframe.copy()
        for feature in dataframe.columns:
            dataframe[feature] = dataframe[feature].apply(
                lambda x: dtype(x.replace(",", ".")) if isinstance(x, str) else dtype(x)
            )
        self.dataframe = dataframe

    def convert_to_float64(self) -> pd.DataFrame:
        self.convert_to_float(self.dataframe, dtype=np.float64)

    def convert_to_float128(self) -> pd.DataFrame:
        self.convert_to_float(self.dataframe, dtype=np.float128)

    def convert_object_or_bool_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert object and bool columns to appropriate numeric types."""
        for label in df.columns:
            dtype_str = str(df[label].dtype)
            if "bool" in dtype_str:
                df[label] = df[label].astype(np.float64)
            elif "object" in dtype_str:
                try:
                    df[label] = df[label].apply(lambda x: float(x.replace(",", ".")) if isinstance(x, str) else float(x)).astype(np.float64)
                except ValueError:
                    df[label] = df[label].astype(str)
        return df

    def _check_for_nan_values(self):
        columns_with_nan_values = self.dataframe.columns[self.dataframe.isnull().any()].tolist()
        if columns_with_nan_values:
            print(f"Warning: Columns {columns_with_nan_values} contain NaN values.")
        else:
            print("DataFrame does not contain NaN values")
        self.dataframe.dropna(inplace=True)

    @property
    def tabular_pd(self) -> TabularPandas:
        """Get a TabularPandas object from the DataFrame."""
        if self.inputs is None or self.outputs is None:
            raise ValueError("Inputs or Outputs are not set!")
        return TabularPandas(self.dataframe, procs=self.procs, cat_names=self.categorical,
                             cont_names=self.continuous, y_names=self.output_names)

    @property
    def input_names(self) -> List[str]:
        return self.categorical + self.continuous

    @property
    def continuous(self) -> List[str]:
        if self.continuous_names is not None:
            return self.continuous_names
        if self.dataframe is None:
            raise ValueError("dataframe is None")
        continuous, _ = custom_cont_cat_split(self.dataframe, max_card=10)
        continuous = [feature for feature in continuous if feature not in self.output_names]
        return continuous

    @property
    def categorical(self) -> List[str]:
        if self.categorical_names is not None:
            return self.categorical_names
        if self.dataframe is None:
            raise ValueError("dataframe is None")
        _, categorical = custom_cont_cat_split(self.dataframe, max_card=10)
        categorical = [feature for feature in categorical if feature not in self.output_names]
        return categorical

    @property
    def num_samples(self) -> int:
        return 0 if self.dataframe is None else self.dataframe.shape[0]

    @property
    def columns(self) -> List[str]:
        return list(self.dataframe.columns)

    @property
    def inputs(self) -> pd.DataFrame:
        if not self.input_names:
            return pd.DataFrame()
        for feature in self.input_names:
            if feature not in self.dataframe.columns:
                raise ValueError(f"Input feature {feature} not in Data!")
            if feature in self.output_names:
                raise ValueError(f"Input feature {feature} also in Output Data!")
        return self.dataframe[self.input_names]

    @property
    def outputs(self) -> pd.DataFrame:
        if not self.output_names:
            return pd.DataFrame()
        for feature in self.output_names:
            if feature not in self.dataframe.columns:
                raise ValueError(f"Output feature {feature} not in Data!")
            if feature in self.input_names:
                raise ValueError(f"Output feature {feature} also in Input Data!")
        return self.dataframe[self.output_names]

    def is_classification(self, name: str = "Class") -> bool:
        """
        Check if the dataset is for classification.

        :param name (str): Name of the target column to check (default is 'Class').
        :returns: bool: True if the target column is for classification, False otherwise.
        """
        if self.dataframe is None:
            raise ValueError("Data is None")
        if name not in self.dataframe.columns:
            return False
        return self.dataframe[name].nunique() < 20

    def normalize(self):
        """Normalize the DataFrame using the DataNormalizer."""
        self.dataframe = self.normalizer.normalize(self.dataframe)
        self.is_normalized = True

    def denormalize(self):
        """Denormalize the DataFrame using the DataNormalizer."""
        self.dataframe = self.normalizer.denormalize(self.dataframe)
        self.is_normalized = False

    def get_balanced_dataset(self, feature: str, feature_vals: List[str], balance_factor: float = 0.5) -> pd.DataFrame:
        """Get a balanced dataset based on the specified feature and balance factor."""
        return self.balancer.get_balanced_dataset(self.dataframe, feature, feature_vals, balance_factor, self.random_seed)

    def replace_inf_by_nan(self) -> None:
        """Replace infinite values with NaN in the DataFrame."""
        self.dataframe = self.dataframe.replace([np.inf, -np.inf], np.nan)

    def drop_nan_entries_and_get_nan_columns(self, selected_features: List[str], verbose: bool = False) -> List[str]:
        """Drop rows with NaN values and return a list of columns with high NaN percentage."""
        drop_columns = []
        nan_percentage = {}
        total_entries = len(self.dataframe)
        for feature in self.dataframe.columns:
            if feature not in selected_features:
                continue
            nan_count = self.dataframe[feature].isna().sum() + self.dataframe[feature].isnull().sum()
            percentage = (nan_count / total_entries) * 100
            nan_percentage[feature] = percentage
            if percentage > 80:
                drop_columns.append(feature)
            else:
                if verbose:
                    print("drop NaN rows for ", feature)
                self.dataframe = self.dataframe.dropna(subset=[feature])
        return drop_columns

    def drop_features(self, features: List[str]) -> None:
        """Drop specified features from the DataFrame."""
        for feature in features:
            self.dataframe.drop([feature], axis=1, inplace=True)

        if self.categorical_names is not None:
            self.categorical_names = [c for c in self.categorical_names if c not in features]

        if self.continuous_names is not None:
            self.continuous_names = [c for c in self.continuous_names if c not in features]

    def remove_samples(self, samples_indices: List[int]) -> None:
        """Remove samples from the DataFrame based on the provided indices."""
        self.dataframe.drop(samples_indices, axis=0, inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

    def onehot_encode(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features in the input DataFrame."""
        inputs_one_hot = inputs.copy()
        for feature in inputs.columns:
            if inputs_one_hot[feature].dtype == np.dtype('object'):
                dummies = pd.get_dummies(inputs_one_hot[feature], prefix=feature)
                inputs_one_hot = pd.concat([inputs_one_hot, dummies], axis=1)
                inputs_one_hot.drop([feature], axis=1, inplace=True)
        return inputs_one_hot

    def get_fastai_data(self, batchsize: int = None) -> Union[TabularPandas, TabularDataLoaders]:
        """Get FastAI's TabularDataLoaders or TabularPandas object for the dataset."""
        tabular_pd = self.tabular_pd
        num_samples_in_dataset = len(tabular_pd)

        if self.mode == "train":
            if batchsize is None:
                raise RuntimeError("Batchsize is not defined!")
            if batchsize > num_samples_in_dataset:
                raise RuntimeError(f"batchsize {batchsize} is bigger than train dataset ({num_samples_in_dataset} samples)!")
            dls = tabular_pd.dataloaders(bs=batchsize)
            dls.rng.seed(self.random_seed)
            return dls

        if self.mode == "valid" and batchsize is not None:
            if batchsize > num_samples_in_dataset:
                raise RuntimeError(f"batchsize {batchsize} is bigger than valid dataset ({num_samples_in_dataset} samples)!")
            dls = tabular_pd.dataloaders(bs=batchsize)
            dls.rng.seed(self.random_seed)
            return dls

        return tabular_pd

    def get_correlation_matrix(self, inputs: pd.DataFrame) -> pd.DataFrame:
        """Compute the correlation matrix for the input DataFrame."""
        inputs = inputs.copy()
        for feature in self.outputs.columns:
            inputs[feature] = self.outputs[feature]
        return inputs.corr()

    def analyze(self) -> None:
        """Analyze and print statistics (mean, std, bounds) of continuous inputs and outputs."""
        def _analyze(data: pd.DataFrame):
            mean, std = data.mean(skipna=True), data.std(skipna=True)
            lower_bounds, upper_bounds = data.min(), data.max()
            for feature in data.columns:
                if lower_bounds[feature] >= upper_bounds[feature]:
                    print(f"WARNING: {feature} has invalid bounds: lower {lower_bounds[feature]} >= upper {upper_bounds[feature]}")
                if std[feature] == 0:
                    print(f"WARNING: {feature}: constant value {mean[feature]:.2f}")
                else:
                    print(f"{feature}: mean={mean[feature]:.2f}, std={std[feature]:.2f}")

        print("Continuous Inputs: ")
        _analyze(self.inputs[self.continuous])
        print("Outputs: ")
        _analyze(self.outputs)

    def reduce_features(self, threshold: float = 0.5) -> None:
        """Reduce features based on correlation with target variables."""
        if self.inputs is None:
            raise RuntimeError("Input is None. Call set_features first!")
        if self.outputs is None:
            raise RuntimeError("Output is None. Call set_targets first!")

        def get_prefix(dummy):
            return dummy.split("_", 1)[0]

        if threshold == -1:
            print("Skip step 'reduce_features': keep all features")
            return

        inputs_one_hot = self.onehot_encode(self.inputs)
        correlation_matrix = self.get_correlation_matrix(inputs_one_hot)
        target_corr = correlation_matrix[self.outputs.columns].drop(self.outputs.columns)

        selected_features = list(
            set(chain.from_iterable([
                key for key, value in target_corr[feature].items() if abs(value) >= threshold and key not in self.outputs.columns
            ] for feature in self.outputs.columns))
        )

        if self.verbose:
            print(f'Found {len(selected_features)} columns with |correlation| >= {threshold}:\n\n{selected_features}.')

        categorical_cols = {col for col in self.categorical if get_prefix(col) in selected_features}
        continuous_cols = {col for col in self.continuous if col in selected_features}

        self.categorical_names = list(categorical_cols)
        self.continuous_names = list(continuous_cols)

        if self.verbose:
            print(f"Reduced input has shape {self.inputs.shape}")

    def reduce_input_features(self, features: List[str]) -> None:
        """Reduce the input features to the specified list of features."""
        self.dataframe = self.dataframe[features + self.output_names]

    def set_normalization_values(self, normalization_values: Dict[str, float]) -> None:
        """Set normalization values for the dataset."""
        self.normalizer.set_normalization_values(normalization_values)

    def save_normalization_values(self, output_folder: str) -> None:
        """Save normalization values to the specified output folder."""
        self.normalizer.save_normalization_values(output_folder)

    def get_normalization_values(self):
        """Get the normalization values."""
        return self.normalizer.normalization_values

    def save(self, filename: str, use_pickle: bool = True):
        """Save the Dataset object to a file."""
        if use_pickle:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        else:
            raise NotImplementedError("Currently, only pickle save is implemented.")

    @classmethod
    def load(cls, filename: str):
        """Load a Dataset object from a file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def to_fastai(self) -> TabularDataLoaders:
        """Convert Dataset to FastAI's TabularDataLoaders."""
        return TabularDataLoaders.from_df(
            self.dataframe, path='.', procs=self.procs, cat_names=self.categorical,
            cont_names=self.continuous, y_names=self.output_names, bs=64
        )
