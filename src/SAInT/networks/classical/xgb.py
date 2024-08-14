from typing import Union, Tuple
from fastai.tabular.all import TabularDataLoaders
from xgboost import XGBRegressor, XGBClassifier
from SAInT.networks.classical.classical import ClassicalModel


class XGBModel(ClassicalModel):

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
                 min_child_weight: int = 1,
                 gamma: float = 0.0,
                 eta: float = 0.3,
                 colsample_bytree: int = 1,
                 subsample: int = 1,
                 n_estimators: int = 100,
                 max_delta_step: float = 0.0,
                 learning_rate: float = 0.1,
                 objective: str = "reg:squarederror",
                 random_seed: int = 123456,
                 model: Union[XGBRegressor, XGBClassifier] = None,
                 verbose: bool = False):
        objective = "binary:logistic" if mode =="classification" else "reg:squarederror"
        if model is None:
            kwargs = {
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "eta": eta,
                "colsample_bytree": colsample_bytree,
                "subsample": subsample,
                "n_estimators": n_estimators,
                "max_delta_step": max_delta_step,
                "learning_rate": learning_rate,
                "objective": objective,
                "random_state": random_seed,
                "eval_metric": "mae"
            }
            model = XGBRegressor(
                **kwargs) if mode == "regression" else XGBClassifier(**kwargs)
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
