import pandas as pd
from fastai.tabular.all import store_attr, Callback
from SAInT.normalization import MeanStdNormalizer, NormalizationMeanStdValues, \
    MinMaxNormalizer, NormalizationMinMaxValues

class OnlineNormalization(Callback):

    def __init__(self):
        super().__init__()
        self.dataframe = None

    def normalize_input_feature(self, conts, feature):
        pass

    def normalize_output_feature(self, conts, feature):
        pass

    def track_data(self, conts, conts_out):
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

    def before_batch(self):
        conts = self.learn.xb[1]
        conts_out = self.learn.yb
        if len(conts_out) > 0:
            conts_out = conts_out[0]
        else:
            conts_out = None

        num_features = conts.shape[1]
        for feature in range(0, num_features):
            conts[:, feature] = self.normalize_input_feature(conts, feature)

        if conts_out is not None:
            num_outputs = conts_out.shape[1]
            for feature in range(0, num_outputs):
                conts_out[:, feature] = self.normalize_output_feature(conts_out, feature)

            if self.data_tracking is True:
                self.track_data(conts, conts_out)
        else:
            print("output empty.")

    def after_fit(self):
        if self.data_tracking is True:
            self.dataframe.to_csv(path_or_buf='data_augmented.csv',
                                  sep=';',
                                  index=False)

class OnlineMeanStdNormalization(OnlineNormalization):

    def __init__(self,
                 input_normalization_values = None,
                 output_normalization_values = None,
                 data_tracking: bool = False,
                 column_input_names: list = None,
                 column_output_names: list = None):
        super().__init__()
        input_mean_values = input_normalization_values.mean_values if input_normalization_values is not None else []
        input_std_values = input_normalization_values.std_values if input_normalization_values is not None else []
        output_mean_values = output_normalization_values.mean_values if output_normalization_values is not None else []
        output_std_values = output_normalization_values.std_values if output_normalization_values is not None else []
        store_attr(input_mean_values=input_mean_values,
                   input_std_values=input_std_values,
                   output_mean_values=output_mean_values,
                   output_std_values=output_std_values,
                   data_tracking=data_tracking,
                   column_input_names=column_input_names,
                   column_output_names=column_output_names)

    def normalize_feature(self, conts, feature, mean_values, std_values):
        return (conts[:, feature] - mean_values.iloc[feature]) / std_values.iloc[feature]

    def normalize_input_feature(self, conts, feature):
        return self.normalize_feature(conts, feature, self.input_mean_values, self.input_std_values)

    def normalize_output_feature(self, conts, feature):
        return self.normalize_feature(conts, feature, self.output_mean_values, self.output_std_values)


class OnlineMinMaxNormalization(OnlineNormalization):

    def __init__(self,
                 input_normalization_values = None,
                 output_normalization_values = None,
                 data_tracking: bool = False,
                 column_input_names: list = None,
                 column_output_names: list = None):
        super().__init__()
        input_min_values = input_normalization_values.min_values if input_normalization_values is not None else []
        input_max_values = input_normalization_values.max_values if input_normalization_values is not None else []
        output_min_values = output_normalization_values.min_values if output_normalization_values is not None else []
        output_max_values = output_normalization_values.max_values if output_normalization_values is not None else []
        store_attr(input_min_values=input_min_values,
                   input_max_values=input_max_values,
                   output_min_values=output_min_values,
                   output_max_values=output_max_values,
                   data_tracking=data_tracking,
                   column_input_names=column_input_names,
                   column_output_names=column_output_names)

    def normalize_feature(self, conts, feature, min_values, max_values):
        return (conts[:, feature] - min_values.iloc[feature]) / (max_values.iloc[feature] - min_values.iloc[feature])

    def normalize_input_feature(self, conts, feature):
        return self.normalize_feature(conts, feature, self.input_min_values, self.input_max_values)

    def normalize_output_feature(self, conts, feature):
        return self.normalize_feature(conts, feature, self.output_min_values, self.output_max_values)
