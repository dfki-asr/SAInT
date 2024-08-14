import random
from typing import List
import pandas as pd


class DataBalancer:
    @staticmethod
    def balance_classes(dataframe: pd.DataFrame, feature: str, feature_vals: List[str], balance_factor: float = 0.5, random_seed: int = 123456) -> List[int]:
        """
        Balance classes in the DataFrame based on a specific feature.

        :param dataframe (pd.DataFrame): Input DataFrame.
        :param feature (str): The feature to balance classes on.
        :param feature_vals (list): List of feature values to balance.
        :param balance_factor (float): The balance factor (default is 0.5).
        :param random_seed (int): Random seed for reproducibility (default is 123456).
        :return: List of selected indices.
        """
        grouped = dataframe.groupby(feature)
        num_items_with_feature_value = grouped.size().to_dict()
        all_indices = grouped.apply(lambda x: x.index.tolist()).to_dict()

        minimum = min(num_items_with_feature_value.values())
        selected = int((float(minimum) / balance_factor) * (1.0 - balance_factor))
        num_selected = {f: min(selected, num_f) for f, num_f in num_items_with_feature_value.items()}
        print(f"{num_items_with_feature_value}, minimum: {minimum}")
        print(f"        select {num_selected}")

        random.seed(random_seed)
        random_indices = [random.choices(all_indices[f], k=num_selected[f]) for f in feature_vals]
        random_indices = sum(random_indices, [])
        return random_indices

    @staticmethod
    def get_balanced_dataset(dataframe: pd.DataFrame, feature: str, feature_vals: List[str], balance_factor: float = 0.5, random_seed: int = 123456) -> pd.DataFrame:
        """Return a balanced dataset based on the specified feature and balance factor."""
        if feature not in dataframe.columns:
            print(f"{feature} not part of Dataframe, use UNBALANCED Dataset!")
            return dataframe

        indices = DataBalancer.balance_classes(dataframe, feature, feature_vals, balance_factor, random_seed=random_seed)
        return dataframe.loc[indices].reset_index(drop=True)
