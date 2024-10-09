from dash import Input, Output
from dash.exceptions import PreventUpdate
from SAInT.dash_application.common.dash_functions import get_pressed_buttons

def register_data_callback(dash_app, app):
    @dash_app.callback(
        Output("sort_criterion_radiobutton", "value"),
        Output("data_radiobutton", "options"),
        Output("data_radiobutton", "value"),
        Output("data_checklist", "options"),
        Output("data_checklist", "value"),
        Output("loss_radiobutton", "value"),
        Output("goodness_of_fit_radiobutton", "value"),
        Output("show_feature_details_radiobutton", "value"),
        Output("update_panel_from_settings", "children"),
        Output("loaded_data", "children"),
        Output("folder_path_info", "value"),
        Output("data-error-dialog", "displayed"),
        Output("data-error-dialog", "message"),
        Input("trigger_load_data_button", "n_clicks"),
        Input("trigger_load_data_after_feature_selection", "children"),
        prevent_initial_call=True
    )
    def update_data(trigger_load_data_button, trigger_load_data_after_feature_selection):
        return _handle_data_update(app, trigger_load_data_button, trigger_load_data_after_feature_selection)

def _handle_data_update(app,
                        trigger_load_data_button,
                        trigger_load_data_after_feature_selection):
    changed_id = get_pressed_buttons()
    error_dialog_displayed = False
    error_dialog_message = ""

    if not "trigger_load_data_after_feature_selection" in changed_id and \
        not "trigger_load_data_button" in changed_id:
        raise PreventUpdate

    data_radiobutton_options = ["train", "valid", "test"]
    sort_criterion_radiobutton_value = "no sorting"
    data_radiobutton_value = ""
    show_datasets_in_plot = ""
    show_feature_details_value = "False"
    loss_radiobutton_value = "mae"
    goodness_of_fit_value = "False"
    update_panel_from_settings = "False"
    loaded_data = "False"
    data_info = ""

    try:
        if "trigger_load_data_after_feature_selection" in changed_id:
            _perform_data_loading(app)

        if app.data_handler is not None:
            if app.application is not None:
                if app.application.trainer is not None:
                    if app.application.trainer.data_settings is not None:
                        loss_radiobutton_value = app.application.trainer.data_settings.metric
                    if app.application.trainer.dataloader is not None:
                        data_info = app.data_handler.get_data_info()
                        dataset_dict = app.application.trainer.dataloader.datasets
                        dataset_names = [mode for mode, dataset in dataset_dict.items() if dataset is not None]
                        data_radiobutton_options = [mode for mode in dataset_names if dataset_dict[mode].num_samples > 0]
                        update_panel_from_settings = "True"
                        loaded_data = "True"
                show_datasets_in_plot = app.application.show_datasets_in_plot
                interactive_plot = app.application.interactive_plot
                if interactive_plot is not None:
                    sort_criterion_radiobutton_value = interactive_plot.sort_criterion
                    data_radiobutton_value = interactive_plot.dataset_selection
                    show_feature_details_value = "True" if interactive_plot.show_feature_details else "False"
                    goodness_of_fit_value = "True" if interactive_plot.goodness_of_fit else "False"
        if show_datasets_in_plot == "":
            show_datasets_in_plot = [data_radiobutton_value]

    except RuntimeError as e:
        error_dialog_displayed = True
        error_dialog_message = str(e)

    return (
        sort_criterion_radiobutton_value,
        data_radiobutton_options,
        data_radiobutton_value,
        data_radiobutton_options,
        show_datasets_in_plot,
        loss_radiobutton_value,
        goodness_of_fit_value,
        show_feature_details_value,
        update_panel_from_settings,
        loaded_data,
        data_info,
        error_dialog_displayed,
        error_dialog_message
    )

def _perform_data_loading(app):
    app.data_handler.load_data()
