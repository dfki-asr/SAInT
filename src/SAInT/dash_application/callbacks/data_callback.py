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
        sort_criterion_radiobutton_value = "no sorting"
        data_radiobutton_value = ""
        show_feature_details_value = "False"
        loss_radiobutton_value = "mae"
        goodness_of_fit_value = "False"
        update_panel_from_settings = "False"
        loaded_data = "False"

        if app.data_handler is not None:
            if app.application is not None:
                interactive_plot = app.application.interactive_plot
                if interactive_plot is not None:
                    sort_criterion_radiobutton_value = interactive_plot.sort_criterion
                    data_radiobutton_value = interactive_plot.dataset_selection
                    show_feature_details_value = "True" if interactive_plot.show_feature_details else "False"
            if app.application.trainer is not None:
                if app.application.trainer.data_settings is not None:
                    loss_radiobutton_value = app.application.trainer.data_settings.metric
                if app.application.trainer.dataloader is not None:
                    dataset_dict = app.application.trainer.dataloader.datasets
                    dataset_names = [mode for mode, dataset in dataset_dict.items() if dataset is not None]
                    data_radiobutton_options = [mode for mode in dataset_names if dataset_dict[mode].num_samples > 0]

                    data_info = app.data_handler.get_data_info()
                    update_panel_from_settings = "True"
                    loaded_data = "True"

        return (
            sort_criterion_radiobutton_value,
            data_radiobutton_options, data_radiobutton_value,
            data_radiobutton_options, [data_radiobutton_value],
            loss_radiobutton_value, goodness_of_fit_value,
            show_feature_details_value,
            update_panel_from_settings, loaded_data, data_info
        )
