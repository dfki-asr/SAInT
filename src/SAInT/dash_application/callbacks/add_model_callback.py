from dash import Input, Output
from SAInT.dash_application.common.config_conversion import settings_value_str_to_dict
from SAInT.dash_application.common.dash_functions import get_pressed_buttons
from SAInT.dash_application.common.path_functions import get_base_model_name

def _get_base_model_name(app):
    if app.application.trainer:
        return get_base_model_name(app.application.trainer.data_folder)
    return ""

def _create_settings_dict(settings_value):
    return settings_value_str_to_dict(settings_value=settings_value)

def _add_model(app, dropdown_value, base_model_name, settings_dict):
    if app.application.model_handler:
        app.application.model_handler.add_model(selected_type=dropdown_value,
                                                    base_model_name=base_model_name,
                                                    settings_dict=settings_dict)

def register_add_model_callback(dash_app, app):
    @dash_app.callback(
        Output("models-configuration-added-models-div", "children"),
        Input("add_model_button", "n_clicks"),
        Input("models-definition-dropdown", "value"),
        Input("models-definition-configuration-input", "value")
    )
    def add_model(n_clicks, dropdown_value, settings_value):
        changed_id = get_pressed_buttons()
        if "add_model_button.n_clicks" in changed_id:
            if app.application.trainer is not None:
                base_model_name = _get_base_model_name(app)
                settings_dict = _create_settings_dict(settings_value)
                _add_model(app, dropdown_value, base_model_name, settings_dict)
            return f"{dropdown_value}: {settings_value}"
        return ""
