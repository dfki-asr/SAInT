from math import floor
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class InteractivePlot:
    def __init__(self, application):
        self.application = application
        self.sort_criterion = "no sorting"
        self.dataset_selection = "train"
        self.goodness_of_fit = False
        self.show_feature_details = True
        self.reset_data()
        self.figure = self.create_figure(application=application)

    def reset(self):
        """Reset plot configuration and data."""
        self.sort_criterion = "no sorting"
        self.dataset_selection = "train"
        self.goodness_of_fit = False
        self.show_feature_details = True
        self.feature_values = {}
        self.y_values = {}
        self.pred_values = {}
        self.original_indices_sorted_values = {}

    def reset_data(self):
        """Reset plot data and internal storage."""
        self.feature_values = {}
        self.y_values = {}
        self.pred_values = {}
        self.original_indices_sorted_values = {}

    def _create_marker_options(self, color, symbol, size=10, line_color='DarkSlateGrey'):
        """Create marker options for scatter plot."""
        return dict(color=color, symbol=symbol, size=size, line=dict(color=line_color, width=1))

    def _create_scatter(self, ds_name, name, x, y, annotations=None, feature_vals=None, marker_options=None, showlegend=True):
        """Create a scatter plot trace."""
        def round_if_float(val, num_digits: int = 3):
            try:
                val = float(val)
            except:
                return val
            return round(val, num_digits) if isinstance(val, float) else val

        def get_feature_text(feature_columns, feature_values):
            if self.show_feature_details:
                feature_text = ", ".join(f"{column}: {round_if_float(val)}" for column, val in zip(feature_columns, feature_values))
                return f"features={feature_text}"
            return ""

        def get_text_for_sample(annotation, feature_columns, feature_values):
            feature_text = get_feature_text(feature_columns, feature_values)
            if feature_text == "":
                return [f"x={annotation}", ds_name]
            return [f"x={annotation}", ds_name, feature_text]

        feature_columns = feature_vals.columns
        text = [", ".join(get_text_for_sample(annotation, feature_columns, feature_values)) for annotation, feature_values in \
                zip(annotations, feature_vals.values)] if annotations else ""
        scatter_function = self._get_scatter_function()
        return scatter_function(mode='markers', x=x, y=y, text=text, name=name, marker=marker_options, showlegend=showlegend)

    def _add_trace(self, figure, row, col, trace):
        """Add a trace to the figure."""
        figure.add_trace(trace, row=row, col=col)

    def _sort_values(self, pred, y, features, sort_idx):
        """Sort values based on the sorting criterion, while keeping track of the original indices."""
        # Store the original indices before sorting
        original_indices = np.arange(len(y))
        if sort_idx == 1:
            sorting_indices = np.argsort(y.values)
        elif sort_idx == 0:
            if pred is None:
                raise RuntimeError("Cannot sort by prediction (no models available!)")
            sorting_indices = np.argsort(pred)
        else:
            return pred, y, features, original_indices

        # Apply the sorting to the original indices
        original_indices_sorted = original_indices[sorting_indices]
        # Sort the y, features, and pred using the sorting indices
        y_sorted = y.iloc[sorting_indices]
        features_sorted = pd.DataFrame(features).iloc[sorting_indices]
        pred_sorted = pred[sorting_indices] if pred is not None else None

        return pred_sorted, y_sorted, features_sorted, original_indices_sorted

    def _create_default_figure(self, height):
        """Create a default empty scatter plot."""
        return px.scatter(pd.DataFrame(), height=height)

    def _get_sort_idx(self):
        """Determine the sorting index based on the sort criterion."""
        return {"by groundtruth": 1, "by prediction": 0}.get(self.sort_criterion, None)

    def _get_predictions(self, application, df):
        """Get predictions from the best model."""
        model = application.model_handler.best_model
        if model:
            _, pred = model.test(df, metric=application.trainer.metric)
            return np.concatenate(pred) if not isinstance(pred, np.ndarray) else pred
        return None

    def _process_predictions(self, pred, output_names, row):
        """Process predictions based on the output names."""
        if len(output_names) > 1:
            return np.array(pred).reshape(-1, len(output_names))[:, row]
        return pred

    def _update_internal_data(self, ds_name, y, features, original_indices_sorted, pred=None):
        """Update internal data storage."""
        if ds_name not in self.y_values:
            self.y_values[ds_name] = []
            self.feature_values[ds_name] = []
            self.pred_values[ds_name] = []
            self.original_indices_sorted_values[ds_name] = []

        self.y_values[ds_name].append(y)
        self.feature_values[ds_name].append(features)
        self.original_indices_sorted_values[ds_name].append(original_indices_sorted)
        if pred is not None:
            self.pred_values[ds_name].append(pred)

    def _calculate_std_error(self, ds_name, y, pred):
        """Calculate the standard error of predictions."""
        error = np.array(y) - np.array(pred)
        return f"{ds_name}: Standard Deviation of Error = {np.std(error)}\n"

    def _create_goodness_of_fit_scatter(self, ds_name, feature_vals, color, symbol, size, x, y, pred, showlegend):
        """Create scatter plot for goodness of fit."""
        marker_options = self._create_marker_options(color=color, symbol=symbol, size=size)
        return self._create_scatter(ds_name=ds_name, name=ds_name, x=pred, y=y,
                                    annotations=x,
                                    feature_vals=feature_vals,
                                    marker_options=marker_options, showlegend=showlegend)

    def _create_groundtruth_scatter(self, ds_name, feature_vals, color, symbol, size, x, y, showlegend):
        """Create scatter plot for ground truth values."""
        marker_options = self._create_marker_options(color=color, symbol=symbol, size=size)
        return self._create_scatter(ds_name=ds_name, name=ds_name + " groundtruth", x=x, y=y,
                                    annotations=x,
                                    feature_vals=feature_vals,
                                    marker_options=marker_options, showlegend=showlegend)

    def _update_min_max_values(self, min_x, max_x, min_y, max_y, pred, y):
        """Update min and max values for the plot axes."""
        min_x = min(min_x, pred.min()) if min_x is not None else pred.min()
        max_x = max(max_x, pred.max()) if max_x is not None else pred.max()
        min_y = min(min_y, y.min()) if min_y is not None else y.min()
        max_y = max(max_y, y.max()) if max_y is not None else y.max()
        return min_x, max_x, min_y, max_y

    def _get_scatter_function(self, max_num_points: int = 50000):
        """Choose scatter function according to the number of points to draw"""
        def count_points(values_dict):
            return sum(len(l) for values in values_dict.values() for l in values if isinstance(values, list))
        num_y_points = count_points(self.y_values)
        num_pred_points = count_points(self.pred_values)
        num_points = num_y_points + num_pred_points
        if num_pred_points > 0 and self.goodness_of_fit:
            # Only half the points, since we draw goodness of fit curve!
            num_points = int(num_points/2)
        print(f"Number of points: {num_points}")
        return go.Scattergl if num_points > max_num_points else go.Scatter

    def _create_optimal_fit_line_trace(self, min_x, min_y, max_x, max_y, color):
        """Create a trace for the optimal fit line."""
        min_point = min(min_x, min_y)
        max_point = max(max_x, max_y)
        scatter_function = self._get_scatter_function()
        return scatter_function(x=[min_point, max_point], y=[min_point, max_point], mode='lines', line=dict(color=color, width=5), showlegend=False)

    def _compute_std_error_across_datasets(self, output_names, dfs, preds, x_values, output_name, row):
        """Compute sum of error deviation over all datasets."""
        if not dfs or not x_values:
            return ""

        std_error_str = ""
        for ds_name, df in dfs.items():
            pred = preds[ds_name]
            if pred is None:
                return ""
            pred_values = self._process_predictions(pred, output_names, row)
            y_values = df[output_name]
            std_error_str += self._calculate_std_error(ds_name, y_values, pred_values)
        return std_error_str

    def _create_subfigure(self, figure, output_names, dfs, preds, x_values, output_name, row, sort_idx, showlegend):
        """Create subfigures for each output name."""
        if not dfs or not x_values:
            return figure, ""

        color_palette = self.application.color_palette.to_rgba_list()
        marker_colors_all = color_palette[:6]
        marker_colors = [marker_colors_all[i] for i in [0, 2, 4]]
        marker_colors_prediction = [marker_colors_all[i] for i in [1, 3, 5]]
        goodness_of_fit_color = color_palette[6]
        marker_symbols = ["circle", "square", "diamond"]
        marker_symbols_prediction = ["cross", "triangle-up", "star"]
        min_x, min_y, max_x, max_y = None, None, None, None
        traces = []
        pixel_def = self.application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        marker_size = pixel_def.marker_size

        for idx, (ds_name, df) in enumerate(dfs.items()):
            x_vals = x_values[ds_name]
            y_vals = df[output_name]
            feature_vals = df[self.application.trainer.dataloader.train.input_names]
            pred = preds[ds_name]

            if pred is None:
                pred_vals, y_vals, feature_vals, original_indices_sorted = self._sort_values(None, y_vals, feature_vals, sort_idx)
                self._update_internal_data(ds_name, y_vals, feature_vals, original_indices_sorted)
                if self.sort_criterion == "by prediction":
                    raise RuntimeError("Cannot sort 'by prediction', since no prediction is given!")
            else:
                pred_vals = self._process_predictions(pred, output_names, row)
                pred_vals, y_vals, feature_vals, original_indices_sorted = self._sort_values(pred_vals, y_vals, feature_vals, sort_idx)
                if not self.goodness_of_fit:
                    marker_options = self._create_marker_options(color=marker_colors_prediction[idx],
                                                                 symbol=marker_symbols_prediction[idx], size=marker_size)
                    scatter_data = self._create_scatter(ds_name=ds_name, name=f"{ds_name} prediction", x=x_vals, y=pred_vals,
                                                        annotations=x_vals,
                                                        feature_vals=feature_vals,
                                                        marker_options=marker_options, showlegend=showlegend)
                    traces.append(scatter_data)

                self._update_internal_data(ds_name, y_vals, feature_vals, original_indices_sorted, pred_vals)

            if self.goodness_of_fit and pred is not None:
                scatter_data = self._create_goodness_of_fit_scatter(ds_name,
                                                                    feature_vals,
                                                                    marker_colors_all[idx], marker_symbols[idx], marker_size, x_vals, y_vals, pred_vals, showlegend)
                traces.append(scatter_data)
                min_x, max_x, min_y, max_y = self._update_min_max_values(min_x, max_x, min_y, max_y, pred_vals, y_vals)
            else:
                scatter_data = self._create_groundtruth_scatter(ds_name,
                                                                feature_vals,
                                                                marker_colors[idx], marker_symbols[idx], marker_size, x_vals, y_vals, showlegend)
                traces.append(scatter_data)

        if max_x is not None:
            line_trace = self._create_optimal_fit_line_trace(min_x, min_y, max_x, max_y, goodness_of_fit_color)
            traces.append(line_trace)
        # Add all collected traces to the figure at once
        for trace in traces:
            figure.add_trace(trace, row=row + 1, col=1)

        return figure

    def create_figure(self, application=None, dfs=None, preds=None, x_vals=None, sort_idx=None, do_save=False):
        """Create the main plot figure."""
        if not application or not dfs:
            return self._create_default_figure(height=476)
        pixel_def = application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error")
        default_height = pixel_def.default_figure_height

        output_names = application.trainer.target_names
        num_rows = len(output_names)
        figure_height = num_rows * default_height
        config = dict(rows=num_rows, cols=1, shared_xaxes=False, shared_yaxes=False, subplot_titles=output_names)
        figure = make_subplots(**config)

        self.reset_data()
        showlegend = True
        for row, output_name in enumerate(output_names):
            figure = self._create_subfigure(figure, output_names, dfs, preds, x_vals, output_name, row, sort_idx, showlegend)
            std_error_str = self._compute_std_error_across_datasets(output_names, dfs, preds, x_vals, output_name, row)
            application.model_handler.std_error_str = std_error_str
            showlegend = False

        fontsize = int(pixel_def.text_font_size.replace("px", ""))
        figure.update_layout(
            clickmode='event+select',
            height=figure_height,
            font=dict(size=fontsize),
            hoverlabel=dict(font=dict(size=fontsize)),
            modebar=dict(
                orientation="h",
                color="black",
                add=["toimage", "zoom", "pan", "zoomin", "zoomout", "autoscale"],
                remove=["lasso", "lasso2d", "select", "select2d"]
            ),
            margin=dict(t=120),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
            ),
            width=pixel_def.figure_width,
            xaxis=dict(showgrid=False, title='', showticklabels=False, showline=False),
            yaxis=dict(showgrid=False, title='', showticklabels=False, showline=False),
            plot_bgcolor='white'
        )

        if not application or not dfs:
            figure.update_xaxes(title='', showticklabels=False, showline=False)
            figure.update_yaxes(title='', showticklabels=False, showline=False)
        else:
            x_axis_text = "Prediction" if self.goodness_of_fit else "Sample"
            y_axis_text = "Groundtruth" if self.goodness_of_fit else "Value"
            figure.update_xaxes(title_text=x_axis_text, showticklabels=True, title_font=dict(size=fontsize))
            figure.update_yaxes(title_text=y_axis_text, showticklabels=True, title_font=dict(size=fontsize))
            if do_save:
                figure.write_image(application.trainer.figure_folder + "/plot.svg")

        return figure

    def update_figure(self, application):
        """Update the plot figure based on the current application state."""
        def get_x_values(df):
            return list(range(df.shape[0]))

        if application.trainer:
            if application.trainer.dataloader:
                data_mode_dict = {
                    "train": application.trainer.dataloader.train,
                    "valid": application.trainer.dataloader.valid,
                    "test": application.trainer.dataloader.test
                }

                sort_idx = self._get_sort_idx()
                dfs, preds, x_vals = {}, {}, {}

                for ds_to_show in application.show_datasets_in_plot:
                    dataset = data_mode_dict.get(ds_to_show)
                    if dataset:
                        dfs[ds_to_show] = dataset.tabular_pd
                        preds[ds_to_show] = self._get_predictions(application, dataset.tabular_pd)
                        x_vals[ds_to_show] = get_x_values(dataset.dataframe)

                self.figure = self.create_figure(application, dfs, preds, x_vals, sort_idx)

    def get_clicked_sample_dict(self, application, selected_data) -> dict:
        """Get data for the clicked sample."""
        def get_output_idx(application, curve_number):
            num_ds = len(application.show_datasets_in_plot)
            curves_per_output = num_ds if self.goodness_of_fit else 2 * num_ds
            return floor(curve_number / curves_per_output)

        point_data = selected_data['points'][0]
        annotation = point_data.get("text", "")
        if annotation:
            splitted = annotation.split(", ")
            print("SPLITTED: ", splitted)
            x_str = splitted[0]
            print("x_str=", x_str)
            ds_name = splitted[1]
            print("ds_name=", ds_name)
        else:
            x_str, ds_name = ("", "")
        sorted_idx = int(x_str.replace("x=", "")) if x_str else None
        curve_number = point_data.get("curveNumber", None)
        output_idx = get_output_idx(application, curve_number)

        output_name = application.trainer.target_names[output_idx]
        dls_train = application.trainer.dataloader.dls_train

        feature_vals = self.feature_values.get(ds_name, [])[output_idx]
        y_vals = self.y_values.get(ds_name, [])[output_idx]
        pred_vals = self.pred_values.get(ds_name, [])[output_idx] if self.pred_values.get(ds_name, []) else None
        original_indices_sorted_vals = self.original_indices_sorted_values.get(ds_name, [])[output_idx] if self.original_indices_sorted_values.get(ds_name, []) else None

        x_val = feature_vals.iloc[sorted_idx] if feature_vals is not None else None
        y_val = y_vals.iloc[sorted_idx] if y_vals is not None else None
        pred_val = pred_vals[sorted_idx] if pred_vals is not None else None
        original_indices_sorted_val = original_indices_sorted_vals[sorted_idx] if original_indices_sorted_vals is not None else None

        return {
            "output_idx": output_idx,
            "output_name": output_name,
            "ds_name": ds_name,
            "dls_train": dls_train,
            "train_data": np.array(dls_train.xs.values, dtype=np.float32),
            "sorted_idx": sorted_idx,
            "original_idx": original_indices_sorted_val,
            "x": x_val,
            "y": y_val,
            "p": pred_val
        }
