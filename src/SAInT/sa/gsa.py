import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from fastai.tabular.all import TabularDataLoaders
from collections import OrderedDict
from SALib import ProblemSpec
from SAInT.model import Model
from SAInT.common import makedirs

class GlobalExplainer:
    def __init__(self, model: Model, data: TabularDataLoaders, num_samples: int = 1024,
                 figure_folder: str = "", do_show: bool = True, do_save: bool = True,
                 random_seed: int = 123456, epsilon: float = 1e-3):
        """
        Initialize the GlobalExplainer with the provided model and data.

        Parameters:
        - model: The model to be explained.
        - data: The data used for explanation.
        - num_samples: Number of samples for analysis.
        - figure_folder: Directory to save figures.
        - do_show: Boolean to control if figures should be shown.
        - do_save: Boolean to control if figures should be saved.
        - random_seed: Seed for random number generation.
        - epsilon: Small value to adjust bounds when lower and upper bounds are the same.
        """
        self.model = model
        self.num_samples = num_samples
        self.figure_folder = figure_folder
        self.do_show = do_show
        self.do_save = do_save
        self.random_seed = random_seed

        lower_bounds = data.xs.min()
        upper_bounds = data.xs.max()
        self._adjust_bounds(lower_bounds, upper_bounds, epsilon)

        output_names = self.model.target_names
        input_names = self.model.input_names
        bounds = [[lower_bounds.iloc[i], upper_bounds.iloc[i]] for i in range(self.model.input_size)]

        self.problem_spec = ProblemSpec({'names': input_names, 'bounds': bounds, 'outputs': output_names})

    def _prediction_function(self, x):
        return self.model.fast_evaluation_fct(x)

    def _adjust_bounds(self, lower_bounds, upper_bounds, epsilon):
        """
        Adjust bounds to ensure lower and upper bounds are not the same.

        Parameters:
        - lower_bounds: The minimum values for each feature.
        - upper_bounds: The maximum values for each feature.
        - epsilon: Small value to adjust bounds.
        """
        for i in range(len(lower_bounds)):
            if lower_bounds.iloc[i] == upper_bounds.iloc[i]:
                feature = lower_bounds.index[i]
                print(f"Warning: Lower and Upper bounds are the same for {feature}: {lower_bounds.iloc[i]}")
                upper_bounds.iloc[i] += epsilon

    def analyze(self, method: str):
        """
        Perform sensitivity analysis using the specified method.

        Parameters:
        - method: The method for analysis ('sobol', 'morris', 'fast', 'rbd-fast').
        """
        num_features = len(self.model.input_names)
        fontsize = 5 if num_features > 20 else 10
        plt.rcParams.update({'font.size': fontsize})
        analysis_method = getattr(self, f"_analyze_{method}", None)
        if analysis_method:
            analysis_method()
        else:
            raise RuntimeError("Invalid method!")

    def _analyze_sobol(self):
        """Perform Sobol sensitivity analysis."""
        self.problem_spec.sample_sobol(self.num_samples, calc_second_order=True).evaluate(
            self._prediction_function).analyze_sobol(
                print_to_console=self.model.verbose, calc_second_order=True, seed=self.random_seed)

    def _analyze_morris(self):
        """Perform Morris sensitivity analysis."""
        self.problem_spec.sample_morris(self.num_samples).evaluate(
            self._prediction_function).analyze_morris(print_to_console=self.model.verbose)

    def _analyze_fast(self):
        """Perform FAST sensitivity analysis."""
        self.problem_spec.sample_fast(self.num_samples).evaluate(
            self._prediction_function).analyze_fast(print_to_console=self.model.verbose, seed=self.random_seed)

    def _analyze_rbd_fast(self):
        """Perform RBD-FAST sensitivity analysis."""
        self.problem_spec.sample_latin(self.num_samples).evaluate(
            self._prediction_function).analyze_rbd_fast(print_to_console=self.model.verbose, seed=self.random_seed)

    def plot(self, title, first_order_color, total_order_color):
        """
        Plot the results of the sensitivity analysis.

        Parameters:
        - title: Title of the plot.
        - first_order_color: Color for first order bars.
        - total_order_color: Color for total order bars.

        Returns:
        - ax: The axis object of the plot.
        """
        ax = self.problem_spec.plot()
        self._replace_plot_colors_and_update_legend_labels(ax, first_order_color, total_order_color)
        plt.xlabel("Feature")
        plt.ylabel("Sobol Value")
        plt.tight_layout()
        self._save_and_show_plot(title)
        return ax

    def _replace_plot_colors_and_update_legend_labels(self, ax, first_order_color, total_order_color):
        """
        Replace plot colors and update legend labels.

        Parameters:
        - ax: The axis object to modify.
        - first_order_color: Color for first order bars.
        - total_order_color: Color for total order bars.
        """
        def annotate_highest_bar_values(axis, legend, bars, top_percent: float = 10.0):
            # Get font size from legend
            legend_fontsize = legend.get_texts()[0].get_fontsize()
            # Get heights of all bars
            heights = np.array([bar.get_height() for bar in bars])
            max_height = np.max(heights)
            # Determine the threshold for the top top_percent %
            threshold = np.percentile(heights, 100.0 - top_percent)
            # Annotate bars that are in the top top_percent %
            for bar in bars:
                height = bar.get_height()
                if height >= threshold:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01 * max_height,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=legend_fontsize
                    )

        def replace_colors_and_annotate_values(axis):
            bars = [patch for patch in axis.patches if isinstance(patch, plt.Rectangle) and patch.get_width() != 0]
            num_features = len(bars) // 2
            new_colors = num_features * [first_order_color] + num_features * [total_order_color]
            for bar, color in zip(bars, new_colors):
                bar.set_color(color)
            from matplotlib.patches import Patch
            legend = axis.legend(handles=[Patch(color=first_order_color, label="First order Sobol"),
                                 Patch(color=total_order_color, label="Total order Sobol")])
            top_percent = 10.0 if num_features > 20 else 100.0
            annotate_highest_bar_values(axis, legend, bars, top_percent)

        if isinstance(ax, np.ndarray):
            for axis in ax:
                replace_colors_and_annotate_values(axis)
        else:
            replace_colors_and_annotate_values(ax)

    def _save_and_show_plot(self, title):
        """
        Save and show the plot based on configuration.

        Parameters:
        - title: Title of the plot for saving.
        """
        if self.do_save:
            makedirs(self.figure_folder)
            plt.savefig(f"{self.figure_folder}{title}.svg", facecolor="white")
        if self.do_show:
            plt.show()

    def get_n_most_important_features(self, factor: float, num_top_features: int = -1) -> dict:
        """
        Get the n most important features from the analysis.

        Parameters:
        - factor: Factor to scale the median value for threshold.
        - num_top_features: Number of top features to return (-1 for all above threshold).

        Returns:
        - dict: Dictionary of most important features per measure.
        """
        measures = {"S1": "first", "mu": "mu", "mu_star": "mu_star", "ST": "total"}
        most_important_features = {}

        for measure, measure_name in measures.items():
            most_important_features[measure_name] = self._get_sorted_features_for_all_targets(measure, factor, num_top_features)

        return most_important_features

    def _get_sorted_features_for_all_targets(self, measure, factor, num_top_features):
        """
        Get sorted features for all targets based on a specific measure.

        Parameters:
        - measure: The measure to sort features by.
        - factor: Factor to scale the median value for threshold.
        - num_top_features: Number of top features to return (-1 for all above threshold).

        Returns:
        - list: List of sorted feature names.
        """
        sorted_features_per_measure = {}
        for target_name in self.model.target_names:
            total_feature_names = self._get_feature_names(target_name)
            sorted_features = self._get_sorted_features_per_measure(measure, target_name, total_feature_names, factor, num_top_features)
            for k, v in sorted_features.items():
                sorted_features_per_measure[k] = max(v, sorted_features_per_measure.get(k, v))

        sorted_features_per_measure = OrderedDict(sorted(sorted_features_per_measure.items(), key=lambda item: item[1], reverse=True))
        if num_top_features > 0:
            sorted_features_per_measure = OrderedDict(list(sorted_features_per_measure.items())[:num_top_features])

        return list(sorted_features_per_measure.keys())

    def _get_feature_names(self, target_name):
        """
        Get feature names from the problem specification.

        Parameters:
        - target_name: The target name to get feature names for.

        Returns:
        - list: List of feature names.
        """
        return self.problem_spec.analysis.get(target_name, {}).get("names", self.problem_spec.get("names", []))

    def _get_sorted_features_per_measure(self, measure, target_name, total_feature_names, factor, num_top_features):
        """
        Get sorted features for a specific measure and target.

        Parameters:
        - measure: The measure to sort features by.
        - target_name: The target name to get features for.
        - total_feature_names: List of all feature names.
        - factor: Factor to scale the median value for threshold.
        - num_top_features: Number of top features to return (-1 for all above threshold).

        Returns:
        - OrderedDict: Sorted features and their values.
        """
        analysis = self.problem_spec.analysis
        values = self._get_measure_values(analysis, measure, target_name)
        sorted_features = self._sort_features_by_importance(values, total_feature_names, factor, num_top_features)
        return sorted_features

    def _get_measure_values(self, analysis, measure, target_name):
        """
        Get values for a specific measure and target.

        Parameters:
        - analysis: The analysis result dictionary.
        - measure: The measure to get values for.
        - target_name: The target name to get values for.

        Returns:
        - np.array: Array of measure values.
        """
        if len(self.model.target_names) > 1:
            return np.array(analysis.get(target_name, {}).get(measure, []))
        else:
            return np.array(analysis.get(measure, []))

    def _sort_features_by_importance(self, values, total_feature_names, factor, num_top_features):
        """
        Sort features by their importance based on measure values.

        Parameters:
        - values: Array of measure values.
        - total_feature_names: List of all feature names.
        - factor: Factor to scale the median value for threshold.
        - num_top_features: Number of top features to return (-1 for all above threshold).

        Returns:
        - OrderedDict: Sorted feature names and their importance values.
        """
        indices = list(range(len(total_feature_names)))
        sorted_dict = OrderedDict(sorted(zip(indices, values), key=lambda item: item[1], reverse=True))
        if num_top_features < 0:
            threshold = factor * np.median(np.abs(values))
            sorted_dict = {k: v for k, v in sorted_dict.items() if np.abs(v) >= threshold}
            if self.model.verbose:
                print(f"Feature importance threshold: {threshold:.5f}")
        sorted_features = OrderedDict({total_feature_names[idx]: value for idx, value in sorted_dict.items()})
        return sorted_features

    def apply_gsa(self, method: str = "fast", factor: float = 1.5,
                  num_top_features: int = -1, colors: list = ["blue", "orange"]) -> dict:
        """
        Apply global sensitivity analysis (GSA) to the model.

        Parameters:
        - method: Method for analysis ('sobol', 'morris', 'fast', 'rbd-fast').
        - factor: Factor to scale the median value for threshold.
        - num_top_features: Number of top features to return (-1 for all above threshold).
        - colors: Colors for first and total order bars.

        Returns:
        - tuple: dictionary of most important features, and axis object of the plot.
        """
        print(f"Method {method}:")
        start = timer()
        self.analyze(method=method)
        end = timer()
        print(f"{method} took {(end - start):.2f} s.")

        if self.model.verbose:
            print("\nSamples: \n", self.problem_spec.samples)
            print("\nResults: \n", self.problem_spec.results)
            print("\nAnalysis: \n", self.problem_spec.analysis)

        gsa_figure = self.plot(title=f"GSA_{method}_N{self.num_samples}", first_order_color=colors[0], total_order_color=colors[1])
        most_important_features = self.get_n_most_important_features(factor, num_top_features)

        for sobol_measure, features in most_important_features.items():
            print(f"\nThe {len(features)} most important features for {sobol_measure} are:\n{features}")

        return most_important_features, gsa_figure
