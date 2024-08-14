from typing import Union, Tuple
import pandas as pd
from fastai.tabular.all import TabularDataLoaders
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.linear_model import Ridge
from SAInT.networks.classical.classical import ClassicalModel


class MultiOutputRidgeModel(ClassicalModel):

    def __init__(self,
                 dls: TabularDataLoaders,
                 name: str,
                 path: str,
                 num_features: int,
                 output_size: int,
                 target_names: list,
                 categorical_cols: list,
                 continuous_cols: list,
                 mode: str,
                 sample_weight: list = None,
                 random_seed: int = 123456,
                 model: Union[MultiOutputRegressor,
                              MultiOutputClassifier] = None,
                 verbose: bool = False):
        if model is None:
            model = MultiOutputRegressor(Ridge(
                random_state=random_seed
            )) if mode == "regression" else MultiOutputClassifier(
                Ridge(random_state=random_seed))
        super().__init__(model,
                         name,
                         path,
                         num_features,
                         output_size,
                         target_names,
                         categorical_cols,
                         continuous_cols,
                         mode,
                         sample_weight=sample_weight,
                         verbose=verbose)
        self.inputs = dls.xs
        self.outputs = dls.ys.values if dls.ys.values.shape[
            1] != 1 else dls.ys.values.ravel()

    def test(self, dls: TabularDataLoaders, metric) -> Tuple[float, list]:
        return super().test_input_and_output(dls.xs, dls.ys.values, metric)

    def get_n_most_important_features(self,
                                      num_top_features: int = 20) -> list:
        all_feature_importances = []
        for i, _ in enumerate(self.model.estimators_):
            all_feature_importances.append(
                pd.DataFrame(self.model.estimators_[i].coef_,
                             index=self.inputs.columns))
        all_feature_importances = pd.concat(
            all_feature_importances).sort_values(by=0, ascending=False)
        self.feature_importances = {
            k: all_feature_importances[0][k][0]
            for k in self.inputs.columns
        }

        top_features = sorted(self.feature_importances,
                              key=lambda x: x[1],
                              reverse=True)[:num_top_features]
        return top_features
