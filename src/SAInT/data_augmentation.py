import numpy as np
import pandas as pd
from fastai.tabular.all import store_attr, Callback
from SAInT.common import random


def rand_sign() -> int:
    return 1 if random.random() < 0.5 else -1


def rand_signs(size: int) -> np.array:
    return np.array([rand_sign() for s in range(0, size)])


class OnlineDataAugmentation(Callback):

    def __init__(self,
                 rel_columns: dict = None,
                 rel_with_thresh_columns: dict = None,
                 epsilon: float = 1e-7,
                 random_seed: int = 123456,
                 data_tracking: bool = False,
                 column_input_names: list = None,
                 column_output_names: list = None):
        super().__init__()
        random.seed(random_seed)
        rel_columns = {} if rel_columns is None else rel_columns
        rel_with_thresh_columns = {} if rel_with_thresh_columns is None else rel_with_thresh_columns
        column_input_names = [] if column_input_names is None else column_input_names
        column_output_names = [] if column_output_names is None else column_output_names
        store_attr(rel_columns=rel_columns,
                   rel_with_thresh_columns=rel_with_thresh_columns,
                   epsilon=epsilon,
                   data_tracking=data_tracking,
                   column_input_names=column_input_names,
                   column_output_names=column_output_names)
        self.dataframe = None

    def before_batch(self):
        conts = self.learn.xb[1]
        conts_out = self.learn.yb
        if len(conts_out) > 0:
            conts_out = conts_out[0]
        else:
            conts_out = None

        batchsize, num_features = conts.shape

        # avoid zero-values (set to epsilon instead)
        for feature in range(0, num_features):
            for idx in range(0, batchsize):
                conts[idx, feature] = conts[
                    idx,
                    feature] if conts[idx, feature] != 0.0 else self.epsilon

        # relative
        for feature, std_value in self.rel_columns.items():
            conts[:, feature] += rand_sign() * random.uniform(
                0, std_value) * conts[:, feature]

        #relative with lower and upper threshold
        for feature, (lower, threshold,
                      upper) in self.rel_with_thresh_columns.items():
            for idx in range(0, batchsize):
                max_std = upper if conts[idx, feature] < threshold else lower
                conts[idx, feature] += rand_sign() * random.uniform(
                    0, max_std) * conts[idx, feature]

        if self.data_tracking is True:
            if conts_out is not None:
                new_in_dataframe = pd.DataFrame(
                    conts.numpy(), columns=self.column_input_names)
                new_out_dataframe = pd.DataFrame(
                    conts_out.numpy(), columns=self.column_output_names)
                new_dataframe = pd.concat(
                    [new_in_dataframe, new_out_dataframe], axis=1)

                if self.dataframe is None:
                    self.dataframe = pd.DataFrame()
                self.dataframe = pd.concat([self.dataframe, new_dataframe],
                                           ignore_index=True)

    def after_fit(self):
        if self.data_tracking is True:
            self.dataframe.to_csv(path_or_buf='data_augmented.csv',
                                  sep=';',
                                  index=False)
