import numpy as np
from typing import Union
import os
import pandas as pd
from fastai.tabular.all import TabularDataLoaders, TabularPandas
from SAInT.dataset import Dataset
from SAInT.data_settings import DataSettings
from SAInT.data_visualizer import DataVisualizer

class DataLoader():
    def __init__(self,
                 train: Dataset = None,
                 valid: Dataset = None,
                 test: Dataset = None,
                 filepath: str = None,
                 output_names: list = None,
                 include_input_features: list = None,
                 exclude_input_features: list = None,
                 delimiter: str = ";",
                 normalization: str = "none",
                 augmented_features: list = None,
                 train_balance_info: str = "original",
                 valid_balance_info: str = "original",
                 test_balance_info: str = "original",
                 balance_function=None,
                 batchsize: int = 8,
                 procs: list = None,
                 do_one_hot_encoding: bool = False,
                 dtype=None,
                 categorical_names: list = None,
                 continuous_names: list = None,
                 verbose: bool = False,
                 kwargs: dict = None):
        super().__init__()
        self.datasets = {
            "train": train if train is not None else None,
            "valid": valid if valid is not None else None,
            "test": test if test is not None else None
        }
        self.train_balance_info = train_balance_info
        self.valid_balance_info = valid_balance_info
        self.test_balance_info = test_balance_info
        self.balance_function = balance_function
        self.batchsize = batchsize
        self.procs = procs
        self.verbose = verbose
        self.kwargs = kwargs if kwargs is not None else {}
        self.categorical_names = categorical_names
        self.continuous_names = continuous_names
        if filepath is not None:
            exclude_input_features = exclude_input_features if exclude_input_features is not None else []
            output_names = output_names if output_names is not None else []
            exclude_input_features = list(exclude_input_features)
            output_names = list(output_names)
            self.load_data(data=filepath,
                           delimiter=delimiter,
                           output_names=output_names,
                           do_one_hot_encoding=do_one_hot_encoding,
                           dtype=dtype)
            self._setup_features(include_input_features, exclude_input_features, output_names, augmented_features, do_one_hot_encoding, normalization)
        self.visualizer = DataVisualizer(self.datasets)

    def create_and_store_subset(self, df, mode, categorical_names, continuous_names):
        self.datasets[mode] = self.create_subdataset(
            dataframe=df,
            mode=mode,
            parent_dataset=self.train,
            categorical_names=categorical_names,
            continuous_names=continuous_names
        )

    def handle_missing_dataset_features(self, dataset_name):
        features = self.train.continuous
        for feature in features:
            if feature not in self.datasets[dataset_name].columns:
                print(f"Need to add {feature} to {dataset_name} dataset!")
                self.datasets[dataset_name].dataframe[feature] = 0

    def _setup_features(self, include_input_features, exclude_input_features, output_names, augmented_features, do_one_hot_encoding, normalization):
        cat = self.datasets["train"].categorical
        cont = self.datasets["train"].continuous
        features = self.datasets["train"].columns

        # Input
        if include_input_features is not None:
            final_include_input_features = self.get_final_selected_features(selected_features=include_input_features,
                                                                        all_features=features,
                                                                        do_one_hot_encoding=do_one_hot_encoding)
            if do_one_hot_encoding:
                cat, cont = [], features
            cat = [f for f in cat if f in final_include_input_features]
            cont = [f for f in cont if f in final_include_input_features]

        # Output
        final_include_output_features = self.get_final_selected_features(selected_features=output_names,
                                                                    all_features=features,
                                                                    do_one_hot_encoding=do_one_hot_encoding)
        output_names = final_include_output_features
        self.train.output_names = output_names

        exclude = output_names + exclude_input_features
        cat = [f for f in cat if f not in exclude]
        cont = [f for f in cont if f not in exclude]
        self.datasets["train"].categorical_names = cat
        self.datasets["train"].continuous_names = cont

        if normalization != "none":
            features_to_normalize = self.train.input_names + self.train.output_names
            if self.verbose:
                print("features_to_normalize: ", features_to_normalize)
            if augmented_features is not None:
                features_to_normalize = [
                    feature for feature in features_to_normalize
                    if feature not in augmented_features
                ]
            self.set_features_to_normalize(features_to_normalize)

    @classmethod
    def from_data_settings(cls, data_settings, dtype=None, procs=None, do_one_hot_encoding=False,
                       valid_frac=0.15, test_frac=0.15, verbose=False):
        def load_dataset_for_mode(mode, subfiles):
            """Helper to choose and load dataset based on mode and data_settings."""
            if len(subfiles) == 1:
                dataset_name = subfiles[0]
            elif data_settings.experiment_type and data_settings.num_total:
                dataset_name = f"data_{data_settings.num_total}_{mode}_{data_settings.experiment_type}.csv"
            else:
                dataset_name = subfiles[0]

            batchsize = int(list(data_settings.batchsize)[0])
            return cls(
                filepath=os.path.join(data_settings.data_folder, dataset_name),
                output_names=data_settings.output_names,
                include_input_features=data_settings.include_input_features,
                exclude_input_features=data_settings.exclude_input_features,
                delimiter=data_settings.delimiter,
                normalization=data_settings.normalization,
                do_one_hot_encoding=do_one_hot_encoding,
                augmented_features=data_settings.augmented_features,
                batchsize=batchsize,
                procs=procs,
                dtype=dtype,
                verbose=data_settings.verbose
            )

        def create_and_store_subset(dataloader, mode, categorical_names, continuous_names):
            """Helper to create and store subset of data for a specific mode."""
            df = dataloaders[mode].train.dataframe
            dataloader.create_and_store_subset(df, mode, categorical_names, continuous_names)

        def handle_missing_data(dataloader, modes, categorical_names, continuous_names):
            """Handle validation and test datasets if they are missing or need processing."""
            for mode in modes:
                if mode in dataloaders:
                    create_and_store_subset(dataloader, mode, categorical_names, continuous_names)
                else:
                    print(f"No {mode} dataset provided.")

        # Main body of from_data_settings
        files = os.listdir(data_settings.data_folder)
        dataloaders = {}

        # Load datasets for each mode
        for mode in data_settings.modi:
            subfiles = [f for f in files if mode in f]
            if not subfiles:
                print(f"No data found for mode {mode}!")
                continue
            dataloader = load_dataset_for_mode(mode, subfiles)
            dataloaders[mode] = dataloader
            print(f"Loaded {dataloader.datasets['train'].dataframe.shape[0]} {mode} samples")

        # Determine primary dataloader and split train/valid/test
        if "total" in dataloaders:
            dataloader = dataloaders["total"]
            dataloader.split_train_valid_test(valid_frac=valid_frac, test_frac=test_frac)
        elif "train" in dataloaders:
            dataloader = dataloaders["train"]
            categorical_names = dataloader.train.categorical
            continuous_names = dataloader.train.continuous
            # Handle validation and test datasets
            handle_missing_data(dataloader, ["valid", "test"], categorical_names, continuous_names)
            # Ensure train dataset is processed
            create_and_store_subset(dataloader, "train", categorical_names, continuous_names)
            dataloader.dataset = None
        else:
            raise RuntimeError("No train dataset available!")

        # Post-processing: Handle NaNs, infinite values, and normalization
        dataloader.replace_inf_by_nan()

        selected_features = dataloader.train.output_names + dataloader.train.categorical + dataloader.train.continuous
        nan_columns = dataloader.drop_nan_entries_and_get_nan_columns(selected_features=selected_features, verbose=verbose)
        dataloader.drop_features(features=nan_columns)

        if data_settings.normalization != "none":
            model_folder = os.path.join(data_settings.output_folder, "models")
            dataloader.normalize_data(output_folder=model_folder)

        # Handle missing features in validation and test datasets
        for mode in ["valid", "test"]:
            if mode in dataloaders:
                dataloader.handle_missing_dataset_features(mode)

        return dataloader

    @staticmethod
    def get_final_selected_features(selected_features, all_features, do_one_hot_encoding) -> list:
        final_features = []
        for f in selected_features:
            if f in all_features:
                final_features.append(f)  # continuous
            elif do_one_hot_encoding:
                # Identify all subfeatures of the categorical feature
                cat_subfeatures = [sf for sf in all_features if sf.startswith(f + "_")]
                if len(cat_subfeatures) > 100:
                    raise RuntimeError(f"Categorical feature '{f}' has more than 100 subfeatures.")
                final_features.extend(cat_subfeatures)
            else:
                print(f"WARNING: Categorical feature '{f}' might not be supported. Consider setting do_one_hot_encoding=True")
        return final_features

    @property
    def train(self) -> Dataset:
        return self.datasets.get("train", None)

    @property
    def valid(self) -> Dataset:
        return self.datasets.get("valid", None)

    @property
    def test(self) -> Dataset:
        return self.datasets.get("test", None)

    @property
    def dls_train(self) -> TabularDataLoaders:
        if self.train is None:
            return None
        train = self.train
        if "balanced" in self.train_balance_info:
            train = self.balance_function(dataloader=self,
                                          balance_info=self.train_balance_info,
                                          mode="train")
        dls_train = train.get_fastai_data(batchsize=self.batchsize)
        if self.valid is not None:
            dls_valid = self.dls_valid
            if dls_valid is not None:
                if len(dls_valid.train) == 0:
                    raise RuntimeError("Could not load validation data!")
                if len(dls_valid.valid) != 0:
                    raise RuntimeError(
                        "Valid part of validation data is not empty")
                dls_train.valid = dls_valid.train
        return dls_train

    @property
    def to_valid(self) -> TabularPandas:
        if self.valid is None:
            return None
        valid = self.valid
        if "balanced" in self.valid_balance_info:
            valid = self.balance_function(dataloader=self,
                                          balance_info=self.valid_balance_info,
                                          mode="valid")
        return valid.get_fastai_data()

    @property
    def dls_valid(self) -> TabularDataLoaders:
        if self.valid is None:
            raise RuntimeError("Validation data is None!")
        return self.valid.get_fastai_data(batchsize=self.batchsize)

    @property
    def to_test(self) -> TabularPandas:
        if self.test is None:
            return None
        test = self.test
        if "balanced" in self.test_balance_info:
            test = self.balance_function(dataloader=self,
                                         balance_info=self.test_balance_info,
                                         mode="valid")
        return test.get_fastai_data()

    @property
    def num_samples(self) -> dict:
        num_samples_dict = {}
        for mode, dataset in self.datasets.items():
            num_samples_dict[
                mode] = dataset.num_samples if dataset is not None else 0
        return num_samples_dict

    def create_subdataset(self,
                          dataframe: pd.DataFrame,
                          mode: str,
                          parent_dataset: Dataset,
                          categorical_names: list = None,
                          continuous_names: list = None):
        return Dataset(
            dataframe=dataframe,
            mode=mode,
            output_names=parent_dataset.output_names,
            normalization_values=parent_dataset.get_normalization_values(),
            features_to_normalize=parent_dataset.normalizer.features_to_normalize,
            normalization=parent_dataset.normalization,
            verbose=parent_dataset.verbose,
            random_seed=parent_dataset.random_seed,
            is_normalized=parent_dataset.is_normalized,
            categorical_names=categorical_names,
            continuous_names=continuous_names,
            procs=parent_dataset.procs)

    def load_data(self,
                  data: Union[str, pd.DataFrame],
                  output_names: list = None,
                  delimiter: str = ",",
                  do_one_hot_encoding: bool = False,
                  dtype=None,
                  chunksize: int = 50000) -> None:
        if self.train is not None:
            raise RuntimeError("Dataset is already loaded!")

        def process_chunk(chunk, do_one_hot_encoding):
            """Helper function to process each chunk."""
            # Replace unwanted characters (avoid inplace to save memory)
            chunk = chunk.replace({',': '.', 'E-0': 'e-'}, regex=True)
            # Convert object columns to category to save memory
            obj_cols = chunk.select_dtypes(include=['object']).columns
            chunk[obj_cols] = chunk[obj_cols].astype('category')
            if do_one_hot_encoding:
                # One-hot encode categorical columns
                categorical_cols = chunk.select_dtypes(include=['object']).columns
                chunk = pd.get_dummies(chunk, columns=categorical_cols, drop_first=False)
            return chunk

        if isinstance(data, str):
            # Reading in chunks to minimize memory footprint
            try:
                chunk_iter = pd.read_csv(data, sep=delimiter, encoding='utf-8', chunksize=chunksize, **self.kwargs)
            except:
                chunk_iter = pd.read_csv(data, sep=delimiter, encoding='ISO-8859-1', chunksize=chunksize, **self.kwargs)

            # Initialize an empty dataframe to concatenate chunks
            df_list = []
            for chunk in chunk_iter:
                # Process each chunk (replace and convert types)
                processed_chunk = process_chunk(chunk, do_one_hot_encoding)
                df_list.append(processed_chunk)
            # Concatenate all processed chunks into a single DataFrame
            csv_df = pd.concat(df_list, ignore_index=True)

            self.datasets["train"] = Dataset(
                dataframe=csv_df,
                mode="train",
                verbose=self.verbose,
                output_names=output_names,
                categorical_names=self.categorical_names,
                continuous_names=self.continuous_names,
                procs=self.procs)
            if dtype == np.float64:
                self.datasets["train"].convert_to_float64()
            if dtype == np.longdouble:
                self.datasets["train"].convert_to_float128()
            if self.verbose:
                print(f"Loaded data from file: {data}.")

        # Check for object types
        if self.datasets['train'] is not None:
            if self.datasets['train'].dataframe is not None:
                dtypes = self.datasets['train'].dataframe.dtypes
                for label, dtype in dtypes.items():
                    if dtype == 'object':
                        print(f"CAUTION: {label}: {dtype}")

        if isinstance(data, pd.DataFrame):
            self.datasets["train"] = Dataset(
                data.copy(),
                verbose=self.verbose,
                output_names=output_names,
                categorical_names=self.categorical_names,
                continuous_names=self.continuous_names,
                procs=self.procs)
            if dtype == np.float64:
                self.datasets["train"].convert_to_float64()
            if dtype == np.longdouble:
                self.datasets["train"].convert_to_float128()

    def check_dropped(self, features: list):
        for f in features:
            if f in self.train.columns:
                raise RuntimeError(
                    f"Drop features error: {f} still in train columns!")
            if self.valid is not None:
                if f in self.valid.columns:
                    raise RuntimeError(
                        f"Drop features error: {f} still in valid columns!")
                if f in self.valid.dataframe.columns:
                    raise RuntimeError(
                        f"Drop features error: {f} still in valid dataframe columns!"
                    )
            if self.test is not None:
                if f in self.test.columns:
                    raise RuntimeError(
                        f"Drop features error: {f} still in test columns!")
                if f in self.test.dataframe.columns:
                    raise RuntimeError(
                        f"Drop features error: {f} still in test dataframe columns!"
                    )
            if f in self.train.dataframe.columns:
                raise RuntimeError(
                    f"Drop features error: {f} still in train dataframe columns!"
                )

    def drop_features(self, features: list):
        if len(features) != 0:
            print(f"Drop features {features}.")
        if self.train is None:
            raise RuntimeError("Dataset is empty! Load data first!")
        self.datasets["train"].drop_features(features)
        if self.valid is not None:
            self.datasets["valid"].drop_features(features)
        if self.test is not None:
            self.datasets["test"].drop_features(features)
        self.filter_features()
        self.check_dropped(features=features)

    def drop_nan_entries_and_get_nan_columns(self, selected_features, verbose: bool = False) -> set:
        if self.train is None:
            raise RuntimeError("Dataset is empty! Load data first!")
        nan_columns = self.train.drop_nan_entries_and_get_nan_columns(selected_features=selected_features, verbose=verbose)
        if self.valid is not None:
            nan_columns += self.valid.drop_nan_entries_and_get_nan_columns(selected_features=selected_features, verbose=verbose)
        if self.test is not None:
            nan_columns += self.test.drop_nan_entries_and_get_nan_columns(selected_features=selected_features, verbose=verbose)
        nan_columns = set(nan_columns)
        return nan_columns

    def replace_inf_by_nan(self) -> None:
        if self.train is None:
            raise RuntimeError("Dataset is empty! Load data first!")
        self.train.replace_inf_by_nan()
        if self.valid is not None:
            self.valid.replace_inf_by_nan()
        if self.test is not None:
            self.test.replace_inf_by_nan()

    def set_features_to_normalize(self, features_to_normalize: list):
        self.train.normalizer.features_to_normalize = features_to_normalize

    def get_balanced_dataset(self,
                             mode: str,
                             feature: str,
                             feature_vals: list,
                             balance_factor: float = 0.5) -> Dataset:
        ds_dict = {"train": self.train, "valid": self.valid, "test": self.test}
        balanced_df = ds_dict[mode].get_balanced_dataset(
            feature=feature,
            feature_vals=feature_vals,
            balance_factor=balance_factor,
            random_seed=ds_dict[mode].random_seed)
        return self.create_subdataset(
            dataframe=balanced_df,
            mode=mode,
            parent_dataset=ds_dict[mode],
            categorical_names=ds_dict[mode].categorical,
            continuous_names=ds_dict[mode].continuous)

    def split_train_valid_test(self,
                               valid_frac: float = 0.2,
                               test_frac: float = 0.1) -> None:
        if self.train.dataframe is None:
            raise RuntimeError("Dataframe is None!")
        print(f"Splitting dataset into {valid_frac} validation and {test_frac} test data...")
        dataframe = self.train.dataframe
        categorical_names = self.train.categorical
        continuous_names = self.train.continuous
        random_seed = self.train.random_seed

        if valid_frac > 0 or test_frac > 0:
            valid_test_frac = valid_frac + test_frac
            # Split data into validation+test and train sets
            valid_test_df = dataframe.sample(frac=valid_test_frac, random_state=random_seed)
            train_df = dataframe.drop(valid_test_df.index)

            # Split valid_test_df into validation and test sets
            valid_size = int(len(valid_test_df) * (valid_frac / valid_test_frac))
            valid_df, test_df = valid_test_df.iloc[:valid_size], valid_test_df.iloc[valid_size:]

            # Create and store datasets
            self.create_and_store_subset(valid_df.copy(), "valid", categorical_names, continuous_names)
            self.create_and_store_subset(test_df.copy(), "test", categorical_names, continuous_names)
            self.create_and_store_subset(train_df.copy(), "train", categorical_names, continuous_names)

            if self.verbose:
                print(f"Split data: {1 - valid_test_frac:.2f} train, {valid_frac:.2f} valid, {test_frac:.2f} test.")
        else:
            # No splitting, use full dataset for training
            self.create_and_store_subset(dataframe.copy(), "train", categorical_names, continuous_names)
            if self.verbose:
                print("No Data Splitting - Training with full dataset.")
        if self.verbose:
            num_train = len(self.datasets.get("train", {}).get('dataframe', []))
            num_valid = len(self.datasets.get("valid", {}).get('dataframe', []))
            num_test = len(self.datasets.get("test", {}).get('dataframe', []))
            print(f"Split data into {num_train} train, {num_valid} valid, and {num_test} test samples.")


    def filter_features(self):
        for mode in self.datasets.keys():
            if self.datasets[mode] is not None:
                cat = self.datasets[mode].categorical
                cont = self.datasets[mode].continuous
                self.datasets[mode].categorical_names = [
                    f for f in cat if f in self.datasets[mode].columns
                ]
                self.datasets[mode].continuous_names = [
                    f for f in cont if f in self.datasets[mode].columns
                ]

    def reduce_features(self, threshold: float = 0.5) -> None:
        if self.train is None:
            raise RuntimeError("Empty dataset!")
        self.datasets["train"].reduce_features(threshold=threshold)
        self.filter_features()

    def normalize_data(self, output_folder: str = None) -> None:
        if self.train is None:
            raise RuntimeError(
                "Data normalization error: No train data available!")
        if self.train.dataframe.shape[0] > 1:
            self.datasets["train"].normalize()

        if output_folder is not None:
            self.datasets["train"].save_normalization_values(
                output_folder=output_folder)

        if self.valid is not None:
            if self.valid.dataframe.shape[0] > 1:
                if self.train.get_normalization_values() is not None:
                    self.datasets["valid"].set_normalization_values(
                        self.train.get_normalization_values())
                self.datasets["valid"].normalize()

        if self.test is not None:
            if self.test.dataframe.shape[0] > 1:
                if self.train.get_normalization_values() is not None:
                    self.datasets["test"].set_normalization_values(
                        self.train.get_normalization_values())
                self.datasets["test"].normalize()

    def reduce_to_features(self, features: list):
        if self.train is not None:
            self.datasets["train"].reduce_input_features(features)
        if self.valid is not None:
            self.datasets["valid"].reduce_input_features(features)
        if self.test is not None:
            self.datasets["test"].reduce_input_features(features)

    def create_histograms(self,
                          figure_folder: str,
                          do_save: bool = True,
                          do_show: bool = True,
                          as_pdf: bool = True):
        self.visualizer.create_histograms(figure_folder, do_save, do_show, as_pdf)

    def analyze(self):
        self.visualizer.analyze()

    def describe(self):
        self.visualizer.describe()

    def display(self):
        self.visualizer.display()


def create_dataloader(data_settings: DataSettings,
                      dtype=None,
                      procs=None,
                      do_one_hot_encoding: bool = False,
                      valid_frac: float = 0.15,
                      test_frac: float = 0.15,
                      verbose: bool = False) -> DataLoader:
    return DataLoader.from_data_settings(data_settings, dtype=dtype, procs=procs,
                                         do_one_hot_encoding=do_one_hot_encoding,
                                         valid_frac=valid_frac, test_frac=test_frac,
                                         verbose=verbose)
