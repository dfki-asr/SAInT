from dash import Input, Output
from SAInT.dash_application.common.dash_functions import get_pressed_buttons
from SAInT.data_settings import save_data_settings_file, json_to_data_settings
from SAInT.dash_application.settings.app_settings import json_to_app_settings, app_settings_to_json
from SAInT.dash_application.common.path_functions import infer_folder_path_from_datasettings_path

def register_load_settings_callback(dash_app, app):
    @dash_app.callback(
        Output("settings-app-json-editor", "value"),
        Output("settings-data-json-editor", "value"),
        Input("loaded_data", "children")
    )
    def load_settings_(children):
        return _handle_load_settings(app)

def register_save_settings_callback(dash_app, app):
    @dash_app.callback(
        Output("saved_to_notification", "children"),
        Input("save_settings_button", "n_clicks"),
        prevent_initial_call=True
    )
    def save_settings(n_clicks):
        return _handle_save_settings(app, n_clicks)

def register_update_app_settings_callback(dash_app, app):
    @dash_app.callback(
        Output("settings-app-json-editor-div", "children"),
        Input("settings-app-json-editor", "value"),
        prevent_initial_call=True
    )
    def update_app_settings_from_json_editor(value):
        return _handle_update_app_settings(app, value)

def register_update_data_settings_callback(dash_app, app):
    @dash_app.callback(
        Output("settings-data-json-editor-div", "children"),
        Input("settings-data-json-editor", "value"),
        prevent_initial_call=True
    )
    def update_data_settings_from_json_editor(value):
        return _handle_update_data_settings(app, value)

def _handle_load_settings(app):
    if "loaded_data.children" in get_pressed_buttons() and app.application.trainer:
        folder_path = app.application.trainer.data_folder
        app_settings_file = f"{folder_path}/app_settings.json"
        app_settings = app.data_handler.load_app_settings(app_settings_file)
        app.application.settings = app_settings
        data_settings = app.application.trainer.data_settings
        settings_app_string = ""
        settings_data_string = ""
        if app_settings:
            settings_app_string = str(app_settings_to_json(app_settings))
        if data_settings:
            settings_data_string = str(app.data_handler.data_settings_to_json(data_settings))
        return settings_app_string, settings_data_string
    return "", ""

def _handle_save_settings(app, n_clicks):
    if "save_settings_button.n_clicks" in get_pressed_buttons() and app.application.trainer:
        folder_path = app.application.trainer.data_folder
        _save_app_settings(app, folder_path)
        _save_data_settings(app, folder_path)
        return ""
    return ""

def _handle_update_app_settings(app, value):
    if "settings-app-json-editor.value" in get_pressed_buttons() and app.application.trainer and value:
        if app.application.settings:
            new_settings = json_to_app_settings(value)
            app.application.settings = new_settings
    return ""

def _handle_update_data_settings(app, value):
    if "settings-data-json-editor.value" in get_pressed_buttons() and app.application.trainer and value:
        if app.application.trainer.data_settings:
            new_settings = json_to_data_settings(value)
            app.application.trainer.data_settings = new_settings
    return ""

def _save_app_settings(app, folder_path):
    settings_file = f"{folder_path}/app_settings.json"
    if app.data_handler:
        app.data_handler.save_app_settings(app.application.settings, settings_file)

def _save_data_settings(app, folder_path):
    settings_file = f"{folder_path}/data_settings.json"
    save_data_settings_file(app.application.trainer.data_settings, settings_file)


def register_data_settings_callback(dash_app, app):
    @dash_app.callback(Output("loaded_data_settings", "children"),
                        Input("select_data_folder_button", "n_clicks"),
                        Input("select_data_file_button", "n_clicks"),
                        prevent_initial_call=True)
    def update_data_settings(n_clicks1, n_clicks2):
        loaded_data_settings = "False"

        changed_id = get_pressed_buttons()
        if "select_data_folder_button.n_clicks" in changed_id:
            folder_path = app.data_handler.select_folder()
            if folder_path:
                app.application = app.data_handler.load_data_settings_and_update_application(folder_path)
                loaded_data_settings = "True"

        if "select_data_file_button.n_clicks" in changed_id:
            data_settings_path = app.data_handler.select_file()
            if data_settings_path:
                folder_path = infer_folder_path_from_datasettings_path(data_settings_path)
                app.application = app.data_handler.load_data_settings_and_update_application(folder_path, data_settings_path)
                loaded_data_settings = "True"

        return loaded_data_settings
