from typing import Tuple
from fastai.tabular.all import TabularDataLoaders
from sklearn.linear_model import GammaRegressor
from SAInT.networks.classical.classical import ClassicalModel


class GammmaRegressorModel(ClassicalModel):

    def __init__(
            self,
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
            alpha: float = 1.0,
            fit_intercept: bool = True,
            max_iter: int = 100,
            tol: float = 0.0001,
            #solver: str = "lbfgs",
            model: GammaRegressor = None,
            verbose: bool = False):
        if verbose:
            gamma_verbose = 1
        else:
            gamma_verbose = 0
        if model is None:
            model = GammaRegressor(
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                #solver=solver,
                verbose=gamma_verbose)
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
