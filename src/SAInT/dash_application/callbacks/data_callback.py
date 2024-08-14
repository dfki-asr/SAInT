from dash import Input, Output
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
        Input("trigger_load_data", "children"),
        prevent_initial_call=True)
    def update_data(children1):
        data_info = ""
        changed_id = get_pressed_buttons()

        if "trigger_load_data.children" in changed_id:
            app.data_handler.load_data()

        data_radiobutton_options = ["train", "valid", "test"]
        if app.application.trainer is not None:
            dataset_dict = app.application.trainer.dataloader.datasets
            dataset_names = [mode for mode, dataset in dataset_dict.items() if dataset is not None]
            data_radiobutton_options = [mode for mode in dataset_names if dataset_dict[mode].num_samples > 0]

        interactive_plot = app.application.interactive_plot
        sort_criterion_radiobutton_value = interactive_plot.sort_criterion or "no sorting"
        data_radiobutton_value = interactive_plot.dataset_selection or ""
        show_feature_details_value = "True" if interactive_plot.show_feature_details else "False"

        loss_radiobutton_value = app.application.trainer.data_settings.metric if app.application.trainer else "mae"
        goodness_of_fit_value = "False"

        update_panel_from_settings = "True"
        loaded_data = "True"
        data_info = app.data_handler.get_data_info()

        return (
            sort_criterion_radiobutton_value,
            data_radiobutton_options, data_radiobutton_value,
            data_radiobutton_options, [data_radiobutton_value],
            loss_radiobutton_value, goodness_of_fit_value,
            show_feature_details_value,
            update_panel_from_settings, loaded_data, data_info
        )
