from typing import Union
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from SAInT.dataset import Dataset
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error
from sklearn.metrics import PredictionErrorDisplay


class DataVisualizer:
    def __init__(self, datasets = None):
        self.datasets = datasets

    @staticmethod
    def cast_data(data: Union[Dataset, pd.DataFrame, list, np.ndarray]) -> pd.DataFrame:
        if isinstance(data, Dataset):
            return data.dataframe
        if isinstance(data, pd.DataFrame):
            return data.astype("float32")
        if isinstance(data, list):
            return pd.DataFrame(np.array(data, np.float32))
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data.astype(np.float32))
        return data

    @staticmethod
    def plot_histogram(data: Union[Dataset, pd.DataFrame, list, np.ndarray],
                       figsize: tuple = None,
                       filepath: str = None,
                       do_save: bool = False,
                       do_show: bool = True,
                       as_pdf: bool = False,
                       same_limits: bool = False) -> None:
        plt.ioff()
        plt.clf()
        data = DataVisualizer.cast_data(data)
        figsize = (data.shape[1], data.shape[1]) if figsize is None else figsize
        filepath = "histogram" if filepath is None else filepath
        fontsize = 8 if len(data.columns) > 10 else 24
        linewidth = 5 if len(data.columns) > 2 else 2
        if len(data.columns) == 1:
            num_rows, num_cols = 1, 1
        elif len(data.columns) == 2:
            num_rows, num_cols = 1, 2
        else:
            num_cols = 5 if len(data.columns) > 6 else 2
            num_rows = int(round(float(len(data.columns)) / float(num_cols) + 0.5))

        fig, fig_ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.subplots_adjust(hspace=0.3, wspace=0.5)
        fig.patch.set_facecolor("white")
        fig_ax = fig_ax.ravel() if num_cols > 1 or num_rows > 1 else [fig_ax]
        xlim_left, xlim_right, ylim_bottom, ylim_top = [], [], [], []
        for i, _ in enumerate(data.columns):
            series = data.iloc[:, i]
            as_hist = not as_pdf
            if as_pdf is True:
                # as Probability Density Function (PDF)
                if series.std() > 1e-7:
                    fig_ax[i] = series.plot(kind='kde', ax=fig_ax[i], grid=False, linewidth=linewidth)
                    fig_ax[i].set_ylabel("Density", fontsize=fontsize)
                else:
                    as_hist = True
            if as_hist is True:
                # Histogram
                fig_ax[i] = series.plot(kind='hist', ax=fig_ax[i], grid=False)
                fig_ax[i].set_ylabel("Frequency", fontsize=fontsize)
            fig_ax[i].set_title(data.columns[i], fontsize=fontsize)
            fig_ax[i].set_xlabel("Value", fontsize=fontsize)
            fig_ax[i].xaxis.set_tick_params(labelsize=fontsize)
            fig_ax[i].yaxis.set_tick_params(labelsize=fontsize)
            xlim_l, xlim_r = fig_ax[i].get_xlim()
            ylim_b, ylim_t = fig_ax[i].get_ylim()
            xlim_left.append(xlim_l)
            xlim_right.append(xlim_r)
            ylim_bottom.append(ylim_b)
            ylim_top.append(ylim_t)

        if same_limits is True:
            for i, _ in enumerate(data.columns):
                fig_ax[i].set_xlim(left=min(xlim_left), right=max(xlim_right))
                fig_ax[i].set_ylim(bottom=min(ylim_bottom), top=max(ylim_top))

        fig.tight_layout()
        if do_save is True:
            fig.savefig(filepath + ".svg", format="svg")
        if do_show is True:
            plt.show()

    @staticmethod
    def pairplot(data: Union[Dataset, pd.DataFrame, list, np.ndarray],
                 figsize: tuple = None,
                 filepath: str = None,
                 do_save: bool = False,
                 do_show: bool = True):
        plt.ioff()
        plt.clf()
        data = DataVisualizer.cast_data(data)
        figsize = (data.shape[1], data.shape[1]) if figsize is None else figsize
        _ = sb.pairplot(data, kind="reg", diag_kind="kde")
        plt.tight_layout()
        if do_save is True:
            plt.savefig(filepath + ".svg", format="svg")
        if do_show is True:
            plt.show()

    @staticmethod
    def plot_pred_err_display(y_test: pd.Series,
                              y_pred_all: dict,
                              figsize: tuple = None,
                              filepath: str = None,
                              do_save: bool = False,
                              do_show: bool = True):

        def compute_score(y_true, y_pred):
            return {
                "R2": f"{r2_score(y_true, y_pred):.3f}",
                "meanAE": f"{mean_absolute_error(y_true, y_pred):.3f}",
                "medianAE": f"{median_absolute_error(y_true, y_pred):.3f}",
            }

        plt.ioff()
        plt.clf()
        figsize = figsize if figsize is not None else (len(y_pred_all) * 6.5, 8)

        f, (ax0, ax1) = plt.subplots(2, len(y_pred_all), sharey="row", figsize=figsize)
        for axis0, axis1, (y_pred_name, y_pred) in zip(ax0, ax1, y_pred_all.items()):
            PredictionErrorDisplay.from_predictions(
                y_test,
                y_pred,
                kind="actual_vs_predicted",
                ax=axis0,
                scatter_kwargs={"alpha": 0.5},
            )
            # Add the score in the legend of each axis
            for name, score in compute_score(y_test, y_pred).items():
                axis0.plot([], [], " ", label=f"{name}={score}")
            axis0.legend(loc="upper left")
            axis0.set_title(f"{y_pred_name}")

            # plot the residuals vs the predicted values
            PredictionErrorDisplay.from_predictions(
                y_test,
                y_pred,
                kind="residual_vs_predicted",
                ax=axis1,
                scatter_kwargs={"alpha": 0.5},
            )
            axis1.set_title(f"{y_pred_name}")

        plt.tight_layout()
        if do_save is True:
            plt.savefig(filepath + ".svg", format="svg")
        if do_show is True:
            plt.show()

    @staticmethod
    def legend_without_duplicate_labels(figure):
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        figure.legend(by_label.values(), by_label.keys())

    @staticmethod
    def get_color_and_label_from_mae(mae_value: float,
                                     mae_thresholds: tuple) -> (str, str):
        if mae_value < mae_thresholds[0]:
            color = "forestgreen"  #"darkgreen"
            label = f" < {mae_thresholds[0]}"
        elif mae_value < mae_thresholds[1]:
            color = "greenyellow"  #"lightgreen"
            label = f"{mae_thresholds[0]}-{mae_thresholds[1]}"
        elif mae_value < mae_thresholds[2]:
            color = "gold"  #"yellow"
            label = f"{mae_thresholds[1]}-{mae_thresholds[2]}"
        elif mae_value < mae_thresholds[3]:
            color = "darkorange"  #"orange"
            label = f"{mae_thresholds[2]}-{mae_thresholds[3]}"
        else:
            color = "red"
            label = f" > {mae_thresholds[3]}"
        return color, label

    @staticmethod
    def plot_values(data: dict,
                    marked_samples: list = None,
                    figsize: tuple = None,
                    filepath: str = None,
                    fontsize: int = 24,
                    markersize: int = 1,
                    linestyle: str = "",
                    draw_lines: bool = False,
                    plot_additional_info: bool = False,
                    mae_thresholds: tuple = None,
                    xlabel: str = "",
                    ylabel: str = "Value",
                    xlim: tuple = None,
                    ylim: tuple = None,
                    do_save: bool = False,
                    do_show: bool = True) -> None:
        plt.ioff()
        plt.clf()
        title = "Prediction vs. Groundtruth"
        suptitle_fs = figsize[0]
        num_rows, num_cols = 1, len(data)

        fig, fig_ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.subplots_adjust(hspace=0.3, wspace=0.5)
        fig.patch.set_facecolor("white")

        fig.suptitle(title, fontsize=suptitle_fs)
        fig_ax = fig_ax.ravel() if num_cols > 1 else [fig_ax]

        mae_values = {}

        for i, _ in enumerate(data.keys()):
            series = data[_]
            y_true = series["y_true"]
            y_pred = series["y_pred"]

            fig_ax[i].scatter(y_true.index, y_true, s=markersize * 2, label="groundtruth", color="blue")
            fig_ax[i].scatter(y_pred.index, y_pred, s=markersize, label="prediction", color="red", linestyle=linestyle)

            if marked_samples is not None:
                for mark in marked_samples:
                    fig_ax[i].axvline(mark, color='r', linestyle='--')

            if draw_lines:
                fig_ax[i].plot([y_true.index, y_pred.index], [y_true.values, y_pred.values], 'k-', linewidth=0.5)

            if plot_additional_info:
                mae = mean_absolute_error(y_true, y_pred)
                mae_values[_] = mae
                color, label = DataVisualizer.get_color_and_label_from_mae(mae, mae_thresholds)
                fig_ax[i].text(
                    0.5, 0.85,
                    f"MAE={mae:.3f}",
                    fontsize=fontsize,
                    color=color,
                    horizontalalignment='center',
                    transform=fig_ax[i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )

            fig_ax[i].set_xlabel(xlabel, fontsize=fontsize)
            fig_ax[i].set_ylabel(ylabel, fontsize=fontsize)
            fig_ax[i].set_title(_, fontsize=fontsize)
            fig_ax[i].xaxis.set_tick_params(labelsize=fontsize)
            fig_ax[i].yaxis.set_tick_params(labelsize=fontsize)

            if xlim is not None:
                fig_ax[i].set_xlim(xlim)
            if ylim is not None:
                fig_ax[i].set_ylim(ylim)

        if plot_additional_info:
            handles, labels = fig_ax[0].get_legend_handles_labels()
            for label in set(labels):
                fig_ax[0].plot([], [], ' ', label=label)
            DataVisualizer.legend_without_duplicate_labels(fig)

        fig.tight_layout()
        if do_save:
            plt.savefig(filepath + ".svg", format="svg")
        if do_show:
            plt.show()

    @staticmethod
    def plot_pred_gt(data: pd.DataFrame,
                    marked_samples: list = None,
                    figsize: tuple = None,
                    filepath: str = None,
                    markersize: int = 1,
                    do_save: bool = False,
                    do_show: bool = True) -> None:
        figsize = (data.shape[1], data.shape[1]) if figsize is None else figsize
        filepath = "histogram" if filepath is None else filepath
        data_sorted = []
        sorted_indices = []
        if len(data.columns) == 2:
            # sort together according to groundtruth
            prediction = data.iloc[:, 0]
            groundtruth = data.iloc[:, 1]
            gt_sorted = groundtruth.sort_values(ascending=True)
            sorted_indices = list(gt_sorted.index)
            pred_vals = [prediction[idx] for idx in sorted_indices]
            # update X from real sample index to chronological (sorted) X
            preds_sorted = pd.Series(data=pred_vals,
                                    index=range(0, len(sorted_indices)))
            gt_sorted = pd.Series(data=gt_sorted.values,
                                index=range(0, len(sorted_indices)))
            data_sorted = {"prediction": preds_sorted, "groundtruth": gt_sorted}
        else:
            raise RuntimeError("Not supported!")

        plot_values(data=data_sorted,
                    marked_samples=marked_samples,
                    figsize=figsize,
                    filepath=filepath,
                    markersize=markersize,
                    do_save=do_save,
                    do_show=do_show)

    @staticmethod
    def plot_correlation_matrix(dataset: Dataset,
                                figsize: tuple = None,
                                filepath: str = "correlation_matrix",
                                do_save: bool = False,
                                do_show: bool = True) -> None:
        plt.ioff()
        plt.clf()
        if do_show is True:
            print(
                f"\nCorrelation matrix of features and targets (last {dataset.outputs.shape[1]} rows):"
            )
        correlation_matrix = dataset.get_correlation_matrix(dataset.dataframe)
        figsize = (correlation_matrix.shape[1],
                correlation_matrix.shape[1]) if figsize is None else figsize
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("white")
        title = filepath.split("/")[-1]
        suptitle_fs = figsize[0]
        fig.suptitle(title, fontsize=suptitle_fs)
        _ = sb.heatmap(round(correlation_matrix, 2),
                    annot=True,
                    cmap="coolwarm",
                    fmt='.2f',
                    square=True,
                    linewidths=.005)
        plt.tight_layout()
        if do_save is True:
            plt.savefig(filepath + ".svg", format="svg")
        if do_show is True:
            plt.show()
