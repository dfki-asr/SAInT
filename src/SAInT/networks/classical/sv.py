from typing import Union, Tuple
from fastai.tabular.all import TabularDataLoaders
from sklearn.svm import SVR, SVC
from sklearn.inspection import permutation_importance
from SAInT.networks.classical.classical import ClassicalModel


class SVModel(ClassicalModel):

    def __init__(self,
                 dls,
                 name: str,
                 path: str,
                 num_features: int,
                 output_size: int,
                 target_names: list,
                 categorical_cols: list,
                 continuous_cols: list,
                 mode: str,
                 sample_weight: list = None,
                 kernel: str = "rbf",
                 degree: int = 3,
                 random_seed: int = 123456,
                 model: Union[SVR, SVC] = None,
                 verbose: bool = False):
        if model is None:
            kwargs = {"kernel": kernel, "degree": degree}
            model = SVR(**kwargs) if mode == "regression" else SVC(**kwargs)
        if dls.ys.values.shape[1] != 1:
            raise RuntimeError(
                "Support vector model does not support multi-target.")
        super().__init__(model, name, path, num_features, output_size,
                         target_names, categorical_cols, continuous_cols, mode,
                         sample_weight, verbose)
        self.inputs = dls.xs
        self.outputs = dls.ys.values.ravel()

    def test(self, dls: TabularDataLoaders, metric) -> Tuple[float, list]:
        return super().test_input_and_output(dls.xs, dls.ys.values.ravel(),
                                             metric)

    def get_n_most_important_features(self,
                                      num_top_features: int = 20) -> list:
        perm_importance = permutation_importance(self.model, self.inputs,
                                                 self.outputs)
        if perm_importance.importances_mean.sum() < 1e-7:
            print(perm_importance)
            raise RuntimeError(
                "Permutation Importance Problem: Output independent of Input!")
        sorted_idx = perm_importance.importances_mean.argsort()
        top_features = list(self.inputs.columns[sorted_idx])[:num_top_features]
        return top_features
