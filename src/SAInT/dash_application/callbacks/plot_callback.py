from dash import html, Input, Output
import matplotlib as mpl
mpl.rcParams["hatch.linewidth"] = 0.3
import matplotlib.pyplot as plt
from SAInT.dash_application.common.dash_functions import get_pressed_buttons
from SAInT.dash_application.common.image_loader import ImageLoader

def register_plot_callback(dash_app, app):
    @dash_app.callback(
        Output("best_model_info", "value"),
        Output("graph", "figure"),
        Output("sort_criterion_radiobutton", "options"),
        Output("goodness_of_fit_radiobutton", "options"),
        Output("error_window_div", "children"),
        Input("compute_errors_button", "n_clicks"),
        Input("get_best_model_button", "n_clicks"),
        Input("sort_criterion_radiobutton", "value"),
        Input("data_radiobutton", "value"),
        Input("data_checklist", "value"),
        Input("goodness_of_fit_radiobutton", "value"),
        Input("show_feature_details_radiobutton", "value"),
        Input("loss_radiobutton", "value"),
        prevent_initial_call=True
    )
    def update_plot(n_clicks1, n_clicks2, sort_criterion, dataset_selection, show_datasets_in_plot, goodness_of_fit, show_feature_details, loss_selection):
        _update_interactive_plot_settings(app, sort_criterion, dataset_selection)
        app.application.show_datasets_in_plot = show_datasets_in_plot if isinstance(show_datasets_in_plot, list) else list(show_datasets_in_plot)
        app.application.interactive_plot.goodness_of_fit = (goodness_of_fit == "True")
        app.application.interactive_plot.show_feature_details = (show_feature_details == "True")
        _update_trainer_metric(app, loss_selection)

        changed_id = get_pressed_buttons()

        if "data_radiobutton.value" in changed_id:
            _handle_data_radiobutton_change(app)
        if "loss_radiobutton.value" in changed_id:
            _handle_loss_radiobutton_change(app)
        if "compute_errors_button.n_clicks" in changed_id:
            _handle_compute_errors_button_click(app)
        if "get_best_model_button.n_clicks" in changed_id:
            _handle_get_best_model_button_click(app)
        if "data_checklist.value" in changed_id or \
            "sort_criterion_radiobutton.value" in changed_id or \
            "goodness_of_fit_radiobutton.value" in changed_id or \
            "show_feature_details_radiobutton.value" in changed_id:
            app.application.interactive_plot.update_figure(app.application)

        error_str, error_window_div_value = _generate_error_window_content(app)
        best_model_info = _get_best_model_info(app)
        sort_criterion_radiobutton_options, goodness_of_fit_radiobutton_options = _get_radiobutton_options(app)

        return error_str + best_model_info, app.application.interactive_plot.figure, \
                sort_criterion_radiobutton_options, goodness_of_fit_radiobutton_options, error_window_div_value

def _update_trainer_metric(app, loss_selection):
    if app.application.trainer is not None:
        if app.application.trainer.data_settings is not None:
            app.application.trainer.data_settings = app.application.trainer.data_settings._replace(metric=loss_selection)

def _handle_data_radiobutton_change(app):
    app.application.model_handler.compute_errors()
    app.application.model_handler.get_best_model()
    app.application.interactive_plot.update_figure(app.application)

def _handle_loss_radiobutton_change(app):
    app.application.model_handler.compute_errors()
    app.application.model_handler.get_best_model()
    app.application.interactive_plot.update_figure(app.application)

def _handle_compute_errors_button_click(app):
    app.application.model_handler.compute_errors()
    app.application.model_handler.get_best_model()
    app.application.interactive_plot.update_figure(app.application)

def _handle_get_best_model_button_click(app):
    app.application.model_handler.get_best_model()
    app.application.interactive_plot.update_figure(app.application)

def _generate_error_window_content(app):
    error_str = ""
    if app.application.trainer is not None:
        error_str = _create_error_plot_and_string(app)
    error_window_div_value = app.application.error_figure
    return error_str, error_window_div_value

def _get_best_model_info(app):
    best_model_info = ""
    if app.application.model_handler.best_model is not None:
        best_model_info = f"\n\nBest model: {app.application.model_handler.best_model.path}"
    return best_model_info

def _get_radiobutton_options(app):
    interactive_plot = app.application.interactive_plot
    sort_criterion_radiobutton_options = ["no sorting", "by groundtruth"]
    if app.application.model_handler.best_model is not None:
        goodness_of_fit_radiobutton_options = ["True", "False"]
        if interactive_plot.goodness_of_fit:
            sort_criterion_radiobutton_options = ["no sorting"]
            app.application.interactive_plot.sort_criterion = sort_criterion_radiobutton_options[0]
        else:
            sort_criterion_radiobutton_options += ["by prediction"]
    else:
        goodness_of_fit_radiobutton_options = ["False"]
    return sort_criterion_radiobutton_options, goodness_of_fit_radiobutton_options

def get_model_type_from_name(model_name):
    if "xgb_" in model_name:
        return "xgb"
    if "rf_" in model_name:
        return "rf"
    if "mlp_" in model_name:
        return "mlp"
    if "res_" in model_name:
        return "res"
    if "tabular_learner_" in model_name:
        return "def"

def get_model_label(model_type, counter_of_type):
    if model_type not in counter_of_type.keys():
        counter_of_type[model_type] = 1
    else:
        counter_of_type[model_type] += 1
    model_label = model_type + "_" + str(counter_of_type[model_type])
    return model_label, counter_of_type

def get_color(color_palette, model_type, assigned_colors):
    if model_type not in assigned_colors.keys():
        idx = len(assigned_colors)
        if idx >= len(color_palette):
            raise RuntimeError(f"Color palette does not have enough colors available for all {idx} model types")
        assigned_colors[model_type] = color_palette[idx]
    color = assigned_colors[model_type]
    return color, assigned_colors

def _create_error_plot_and_string(app):
    # The implementation for create_error_plot_and_string would go here
    if app.application.trainer is None:
        return ""
    errors = app.application.trainer.errors
    selected = app.application.interactive_plot.dataset_selection
    if selected not in errors.keys():
        return ""
    if errors[selected] == {}:
        return ""
    metric = app.application.trainer.metric
    figure_folder = app.application.trainer.figure_folder

    def get_errors_and_models_as_lists(model_names):
        color_palette_list = app.application.color_palette.to_normalized_rgba_list()
        errors_list, models_list, colors_list = [], [], []
        error_str = "Error:"
        counter_of_type, assigned_colors = {}, {}
        for model_name in model_names:
            if selected in errors.keys():
                if metric in errors[selected].keys():
                    if model_name in errors[selected][metric].keys():
                        error_of_model_on_selected_dataset = errors[selected][metric][model_name]
                        model_type = get_model_type_from_name(model_name)
                        model_label, counter_of_type = get_model_label(model_type, counter_of_type)
                        color, assigned_colors = get_color(color_palette_list, model_type, assigned_colors)
                        error_str += f"\n{model_label}:   name={model_name},\n   {selected} data: {metric}: {error_of_model_on_selected_dataset:.5f}"
                        errors_list.append(error_of_model_on_selected_dataset)
                        models_list.append(model_label)
                        colors_list.append(color)
        return errors_list, models_list, error_str, colors_list

    def create_bar_plot(models_list, errors_list, color_list, metric, figure_folder):
        def calculate_fontsize(x, min_items=1, max_items=20, min_fontsize=16, max_fontsize=33):
            # Ensure x is within the bounds
            x = max(min_items, min(max_items, x))
            # Linear interpolation of font size
            fontsize = max_fontsize - (max_fontsize - min_fontsize) * ((x - min_items) / (max_items - min_items))
            return int(fontsize)

        pixel_def = app.application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        text_font_size = pixel_def.text_font_size
        error_plot_width = pixel_def.error_plot_width
        error_plot_height = pixel_def.error_plot_height
        plt.figure(figsize=(16, 8))
        # Define hatches to differentiate bars
        hatches = ['/', 'o', '-', '\\', '.', '+', 'x', 'O', '|', '*']
        # Create a mapping of colors to hatches
        unique_colors = list(set(color_list))
        color_to_hatch = {color: hatches[i % len(hatches)] for i, color in enumerate(unique_colors)}
        # Plot bars with colors and corresponding hatches
        bars = plt.bar(models_list, errors_list, color=color_list, edgecolor='black')
        # Apply hatches based on color
        for bar, color in zip(bars, color_list):
            bar.set_hatch(color_to_hatch[color])
        fontsize = 2.5 * int(text_font_size.replace("px", ""))
        plt.xlabel("Model", fontsize=fontsize)
        adaptive_fontsize = calculate_fontsize(x=len(models_list), max_fontsize=fontsize)
        plt.xticks(fontsize=adaptive_fontsize)
        plt.yticks(fontsize=adaptive_fontsize)
        metric_str = str(metric).upper()
        plt.ylabel(f"{metric_str}", fontsize=fontsize)
        plt.title(f"{metric_str} of models on {selected} dataset", fontsize=fontsize)
        figure_path = figure_folder + "/error_on_models.svg"
        plt.savefig(figure_path, facecolor="white")
        image_loader = ImageLoader()
        src = image_loader.load_svg_from_file(figure_path, width=error_plot_width, height=error_plot_height)
        return src

    model_names = list(app.application.trainer.models.keys())
    errors_list, models_list, error_str, color_list = get_errors_and_models_as_lists(model_names=model_names)
    src = create_bar_plot(models_list, errors_list, color_list, metric, figure_folder)
    app.application.error_figure = [
        html.Img(src=src, width="50%")
    ]
    return error_str

def _update_interactive_plot_settings(app, sort_criterion, dataset_selection):
    app.application.interactive_plot.sort_criterion = sort_criterion
    app.application.interactive_plot.dataset_selection = dataset_selection
