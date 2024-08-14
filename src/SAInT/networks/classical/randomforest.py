from typing import Union, Tuple
from fastai.tabular.all import TabularDataLoaders
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from SAInT.networks.classical.classical import ClassicalModel


class RandomForestModel(ClassicalModel):

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
                 max_depth: int = 30,
                 n_estimators: int = 100,
                 max_features: str = None,
                 random_seed: int = 123456,
                 model: Union[RandomForestRegressor,
                              RandomForestClassifier] = None,
                 verbose: bool = False):
        if model is None:
            kwargs = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features,
                "random_state": random_seed
            }
            model = RandomForestRegressor(
                **kwargs) if mode == "regression" else RandomForestClassifier(
                    **kwargs)
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
        self.feature_importances = dict(
            zip(self.inputs.columns, self.model.feature_importances_))

        top_features = sorted(self.feature_importances,
                              key=lambda x: x[1],
                              reverse=True)[:num_top_features]
        return top_features
