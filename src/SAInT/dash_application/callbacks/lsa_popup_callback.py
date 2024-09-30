from pandas import DataFrame
from dash import Input, Output
from SAInT.dash_application.common.dash_functions import get_pressed_buttons
from dash import html

def register_lsa_popup_callback(dash_app, app):
    @dash_app.callback(
        Output("lsa_popup", "is_open"),
        Output("lsa_window_popup_div", "children"),
        Output("graph", "clickData"),
        Output("norm_denorm_radiobutton", "value"),
        Input("graph", "clickData"),
        Input("close_lsa_popup", "n_clicks"),
        Input("norm_denorm_radiobutton", "value"),
        prevent_initial_call=True
    )
    def update_lsa_popup(selected_data, n_clicks_lsa_popup, norm_denorm_value):
        changed_id = get_pressed_buttons()
        if "graph.clickData" in changed_id and selected_data is not None:
            _handle_graph_click(app, selected_data)
            norm_denorm_value = _get_norm_denorm_value(app)
        if "close_lsa_popup.n_clicks" in changed_id:
            app.application.lsa_popup.close()
        if "norm_denorm_radiobutton.value" in changed_id:
            _update_norm_denorm(app, selected_data, norm_denorm_value)

        return app.application.lsa_popup.is_open, app.application.lsa_popup.get_content(), selected_data, norm_denorm_value

def _handle_graph_click(app, selected_data):
    sample_dict = app.application.interactive_plot.get_clicked_sample_dict(app.application, selected_data)
    app.application.local_explainer.explain(sample_dict)

def _get_norm_denorm_value(app):
    selected_dataset = app.application.interactive_plot.dataset_selection
    is_normalized = app.application.trainer.dataloader.datasets[selected_dataset].is_normalized
    return "normalized" if is_normalized else "denormalized"


def _update_norm_denorm(app, selected_data, norm_denorm_value):
    if norm_denorm_value == "denormalized":
        _denormalize_data(app, selected_data)
    else:
        _set_lsa_popup_content(app, selected_data)


def _denormalize_data(app, selected_data):
    sample_dict = app.application.interactive_plot.get_clicked_sample_dict(app.application, selected_data)
    x, y, p = sample_dict["x"], sample_dict["y"], sample_dict["p"]
    selected_dataset = app.application.interactive_plot.dataset_selection
    normalizer = app.application.trainer.dataloader.datasets[selected_dataset].normalizer
    selected_features = normalizer.features_to_normalize
    target_name = sample_dict["output_name"]

    y_denorm, p_denorm, x_denorm_text = _get_denormalized_values(normalizer, selected_features, target_name, x, y, p)
    pred_mae_text = app.application.local_explainer._generate_prediction_info(y_denorm, p_denorm)

    window = app.application.lsa_popup.window
    window.children[1] = html.P(f"groundtruth: {y_denorm:.5f}{pred_mae_text}")
    window.children[-1] = html.P(f"{x_denorm_text}")
    app.application.lsa_popup.window = window


def _get_denormalized_values(normalizer, selected_features, target_name, x, y, p):
    y_denorm = _denormalize_single_value(normalizer, selected_features, target_name, x, y)
    p_denorm = _denormalize_single_value(normalizer, selected_features, target_name, x, p)
    x_denorm = _denormalize_features(normalizer, selected_features, x)
    x_denorm_text = ", ".join([f"{k}: {v[0]}" for k, v in x_denorm.items()])
    return y_denorm, p_denorm, x_denorm_text


def _denormalize_single_value(normalizer, selected_features, target_name, x, value):
    len_y = len(selected_features) - len(x.values)
    all_y_names = selected_features[-len_y:]
    tmp = [0.0 for _ in range(len_y)]
    y_idx = all_y_names.index(target_name)
    tmp[y_idx] = value
    complete_data = [list(x.values) + tmp]
    data_normalized = DataFrame(data=complete_data, columns=selected_features)
    data_denormalized = normalizer.denormalize(data_normalized)
    normalizer.is_normalized = True
    return data_denormalized[target_name].iloc[0]


def _denormalize_features(normalizer, selected_features, x):
    data_normalized = DataFrame(data=[list(x.values)], columns=selected_features[:len(x.values)])
    normalizer.features_to_normalize = selected_features[:len(x.values)]
    data_denormalized = normalizer.denormalize(data_normalized)
    normalizer.is_normalized = True
    normalizer.features_to_normalize = selected_features
    return data_denormalized


def _set_lsa_popup_content(app, selected_data):
    sample_dict = app.application.interactive_plot.get_clicked_sample_dict(app.application, selected_data)
    x, y, p = sample_dict["x"], sample_dict["y"], sample_dict["p"]
    pred_mae_text = app.application.local_explainer._generate_prediction_info(y, p)
    window = app.application.lsa_popup.window
    window.children[1] = html.P(f"groundtruth: {y:.5f}{pred_mae_text}")
    features = ", ".join([f"{k}: {v}" for k, v in sample_dict["x"].items()])
    window.children[-1] = html.P(f"{features}")
    app.application.lsa_popup.window = window
