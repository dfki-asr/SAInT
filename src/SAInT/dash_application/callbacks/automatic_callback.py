from dash import Input, Output
from SAInT.dash_application.common.dash_functions import get_pressed_buttons

def register_auto_load_data_folder_callback(dash_app, app):
    @dash_app.callback(Output("select_data_folder_button", "n_clicks"),
                        [Input("select_data_folder_button", "n_clicks")],
                        prevent_initial_call=False)
    def auto_load_data_folder(n_clicks):
        import time
        time.sleep(2)
        return 1

def register_auto_compute_error_callback(dash_app, app):
    @dash_app.callback(Output("compute_errors_button", "n_clicks"),
                        Input("models_info", "value"),
                        prevent_initial_call=True)
    def auto_compute_errors(models_info_value):
        changed_id = get_pressed_buttons()
        if "models_info.value" in changed_id:
            return 1
        return 0

def register_auto_gsa_callback(dash_app, app):
    @dash_app.callback(Output("gsa_best_model_button", "n_clicks"),
                        Input("best_model_info", "value"),
                        prevent_initial_call=False)
    def auto_perform_gsa(best_model_info_value):
        changed_id = get_pressed_buttons()
        if "best_model_info.value" in changed_id:
            if app.application.model_handler.best_model is not None:
                if app.application.model_handler.best_model_was_changed:
                    return 1
                return 0
            return 0
        return 0

def register_auto_reloaddata_callback(dash_app, app):
    @dash_app.callback(Output("trigger_load_data_button", "n_clicks"),
                        Input("gsa_best_model_button", "n_clicks"),
                        Input("screen-dimensions-div", "children"),
                        prevent_initial_call=False)
    def auto_perform_reloaddata(n_clicks, screendims):
        changed_id = get_pressed_buttons()
        if "screen-dimensions-div.children" in changed_id:
            app._setup_layout()
        if "gsa_best_model_button.n_clicks" in changed_id or "screen-dimensions-div.children" in changed_id:
            if app.application.trainer is not None:
                if app.application.trainer.dataloader is not None:
                    return 1
                return 0
            return 0
        return 0
