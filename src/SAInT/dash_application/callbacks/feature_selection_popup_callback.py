from dash import Input, Output
from SAInT.dash_application.common.dash_functions import get_pressed_buttons

def register_close_feature_selection_popup_callback(dash_app, app):
    @dash_app.callback(
        Output("trigger_load_data_after_feature_selection", "children"),
        Input("save_outputnames_button", "n_clicks"),
        Input("close_feature_selection_popup", "n_clicks"),
        prevent_initial_call=True
    )
    def update_close_feature_selection_popup(n_clicks_save, n_clicks_close):
        return _handle_close_feature_selection_popup()

def _handle_close_feature_selection_popup():
    changed_id = get_pressed_buttons()
    if "save_outputnames_button.n_clicks" in changed_id or "close_feature_selection_popup.n_clicks" in changed_id:
        return "True"
    return "False"

def register_open_feature_selection_popup_callback(dash_app, app):
    @dash_app.callback(
        Output("outputnames_window_popup_div", "children"),
        Output("feature_selection_popup", "is_open"),
        Output("feature-error-dialog", "displayed"),
        Output("feature-error-dialog", "message"),
        Input("loaded_data_settings", "children"),
        Input("save_outputnames_button", "n_clicks"),
        Input("close_feature_selection_popup", "n_clicks"),
        Input("outputnames_window_popup_div", "children"),
        prevent_initial_call=True
    )
    def update_open_feature_selection_popup(children1, n_clicks_save, n_clicks_close, children2):
        return _handle_open_feature_selection_popup(app, children1, n_clicks_save, n_clicks_close, children2)

def _handle_open_feature_selection_popup(app, children1, n_clicks_save, n_clicks_close, children2):
    changed_id = get_pressed_buttons()
    error_dialog_displayed, error_dialog_message = False, ""
    try:
        if "loaded_data_settings.children" in changed_id:
            _handle_loaded_data_settings_change(app)
        if "save_outputnames_button.n_clicks" in changed_id:
            _handle_save_outputnames_button_click(app, children2)
        if "close_feature_selection_popup.n_clicks" in changed_id:
            _handle_close_feature_selection_popup_click(app)
    except RuntimeError as e:
        error_dialog_displayed = True
        error_dialog_message = str(e)
    feature_selection_popup_is_open = app.application.feature_selection_popup.is_open
    outputnames_window_popup_div = app.application.feature_selection_popup.get_content()
    return outputnames_window_popup_div, feature_selection_popup_is_open, error_dialog_displayed, error_dialog_message

def _handle_loaded_data_settings_change(app):
    if app.application.trainer:
        app.data_handler.open_feature_selection_popup()

def _handle_save_outputnames_button_click(app, children2):
    popup_body = children2["props"]["children"]
    app.data_handler.save_to_data_settings_file(popup_body)
    app.application.feature_selection_popup.close()

def _handle_close_feature_selection_popup_click(app):
    app.application.feature_selection_popup.close()
