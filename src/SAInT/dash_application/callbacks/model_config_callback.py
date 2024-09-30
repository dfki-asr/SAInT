from dash import Input, Output
from SAInT.dash_application.common.dash_functions import get_pressed_buttons
from SAInT.dash_application import saint_dash_layout_components as layout

def _get_setting_for_selected_model(app, value):
    settings = app.application.settings
    trainer_data_settings = app.application.trainer.data_settings if app.application.trainer else None
    
    if value == "xgb_model_selected":
        xgboost_depth = settings.xgb_max_depth
        xgboost_n_estimators = settings.xgb_n_estimators
        setting = f"max_depth={xgboost_depth}, n_estimators={xgboost_n_estimators}"
    elif value == "rf_model_selected":
        rf_depth = settings.rf_max_depth
        rf_n_estimators = settings.rf_n_estimators
        setting = f"max_depth={rf_depth}, n_estimators={rf_n_estimators}"
    elif value == "mlp_model_selected":
        if trainer_data_settings:
            batchsize = trainer_data_settings.batchsize
            hidden_layers = settings.mlp_hidden_layers
            mlp_dropout = settings.mlp_dropout
            max_epochs = settings.max_epochs
            patience = settings.patience
            setting = f"hidden_layers={hidden_layers}, dropout={mlp_dropout}, batchsize={batchsize}, max_epochs={max_epochs}, patience={patience}"
        else:
            setting = ""
    elif value == "resnet_model_selected":
        if trainer_data_settings:
            batchsize = trainer_data_settings.batchsize
            resnet_layersize = settings.resnet_layersizes
            resnet_num_blocks = settings.resnet_blocks
            resnet_dropout = settings.resnet_dropout
            max_epochs = settings.max_epochs
            patience = settings.patience
            setting = f"layersize={resnet_layersize}, num_blocks={resnet_num_blocks}, dropout={resnet_dropout}, batchsize={batchsize}, max_epochs={max_epochs}, patience={patience}"
        else:
            setting = ""
    else:
        setting = ""
    return setting

def _create_add_button(setting, pixel_def):
    if setting:
        return layout.create_icon_button(label="Add", class_name="fa fa-plus", id="add_model_button").to_html(pixel_def)
    return ""

def register_model_configuration_callback(dash_app, app):
    @dash_app.callback(
        Output("models-definition-configuration-input", "value"),
        Output("models-configuration-button-div", "children"),
        Input("models-definition-dropdown", "value")
    )
    def set_model_configuration_input(value):
        changed_id = get_pressed_buttons()
        if "models-definition-dropdown.value" in changed_id:
            if app.application.settings is not None:
                setting = _get_setting_for_selected_model(app, value)
                pixel_def = app.application.pixel_definitions
                add_button = _create_add_button(setting, pixel_def)
                return setting, add_button
        return "", ""
