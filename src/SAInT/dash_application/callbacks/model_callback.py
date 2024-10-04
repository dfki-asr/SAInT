from dash import Input, Output
from dash.exceptions import PreventUpdate
from SAInT.dash_application.common.dash_functions import get_pressed_buttons

def register_model_loading_callback(dash_app, app):
    @dash_app.callback(
        Output("models_loaded_info", "children"),
        Input("load_models_button", "n_clicks"),
        Input("load_model_button", "n_clicks"),
        Input("sort_criterion_radiobutton", "value"),
        Input("data_radiobutton", "value"),
        prevent_initial_call=True
    )
    def update_models(n_clicks1, n_clicks2, sort_criterion_radiobutton, data_radiobutton):
        _update_interactive_plot_settings(app, sort_criterion_radiobutton, data_radiobutton)
        changed_id = get_pressed_buttons()
        
        if "load_models_button.n_clicks" in changed_id:
            app.application.model_handler.load_models()
        elif "load_model_button.n_clicks" in changed_id:
            app.application.model_handler.load_model()

        return "" if app.application.trainer is None else "models loaded"

def register_model_loaded_or_trained_callback(dash_app, app):
    @dash_app.callback(
        Output("models_info", "value"),
        Input("models_trained_info", "children"),
        Input("models_loaded_info", "children"),
        Input("models-configuration-added-models-div", "children"),
        Input("sort_criterion_radiobutton", "value"),
        Input("data_radiobutton", "value"),
        Input("loaded_data", "children"),
        prevent_initial_call=True
    )
    def update_models(model_trained, model_loaded, model_added, sort_criterion_radiobutton, data_radiobutton, loaded_data):
        _update_interactive_plot_settings(app, sort_criterion_radiobutton, data_radiobutton)

        if app.application.trainer is None:
            return ""

        model_names_str = "\n".join(app.application.trainer.models.keys())
        if len(app.application.trainer.models) == 0:
            return ""
        models_info = f"{len(app.application.trainer.models)} Models: {app.application.trainer.model_folder}:\n\n{model_names_str}"
        return models_info


def _update_interactive_plot_settings(app, sort_criterion, dataset_selection):
    app.application.interactive_plot.sort_criterion = sort_criterion
    app.application.interactive_plot.dataset_selection = dataset_selection
