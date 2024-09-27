from SAInT.dash_application.components import *
from SAInT.metric import get_list_of_supported_standard_metrics

def create_icon_button(label, class_name, id):
    return DashIconButton(label=label, class_name=class_name, id=id)

def create_console():
    return DashConsole()

def create_gsa_window():
    return DashWindow(default_value="", id="gsa_window_div")

def create_error_window():
    return DashWindow(default_value="", id="error_window_div")

def create_invisible_div(id):
    return DashDiv(id=id, content=[], visible=False)

def create_div(id, content, width: str = None, margin: str = None):
    return DashDiv(id=id, content=content, width=width, margin=margin, visible=True)

def create_link(rel, href):
    return DashLink(rel=rel, href=href)

def create_graph(figure, id: str):
    return DashGraph(figure=figure, id=id)

def create_popup(title: str, id_popup: str, id_window: str, id_close: str, id_save: str = None, fullscreen: bool = True):
    return DashPopup(title=title, id_popup=id_popup, id_window=id_window,
                     id_close=id_close, id_save=id_save, fullscreen=fullscreen)

def create_header(title, logo):
    return DashHeader(title=title, logo=logo)

def create_visualization_panel(default_selected_dataset, default_sort_criterion, default_goodness_of_fit, default_show_feature_details):
    data_checklist = DashChecklist(name="Show Dataset", options=["train", "valid", "test"],
                                   default_value=default_selected_dataset, id="data_checklist")
    sort_criterion_radio_buttons = DashRadioButton(name="Sort Samples",
                                                   options=["no sorting", "by groundtruth"],
                                                   default_value=default_sort_criterion,
                                                   id="sort_criterion_radiobutton")
    goodness_of_fit_radio_buttons = DashRadioButton(name="Goodness of fit",
                                                    options=["False"],
                                                    default_value=default_goodness_of_fit,
                                                    id="goodness_of_fit_radiobutton")
    show_feature_details_radio_buttons = DashRadioButton(name="Feature Details",
                                                    options=["False", "True"],
                                                    default_value=default_show_feature_details,
                                                    id="show_feature_details_radiobutton")
    return DashGrid(item_values=[
        DashHeading(title="Visualization", id="visualization-heading"),
        data_checklist,
        sort_criterion_radio_buttons,
        goodness_of_fit_radio_buttons,
        show_feature_details_radio_buttons
        ],
        item_widths=[2, 2, 3, 2, 2], id="visualization_panel")

def create_explanation_panel():
    explain_checklist = DashChecklist(name="Explain with", options=["lime", "shap"],
                                      default_value=["lime", "shap"], id="explain_checklist")
    return DashGrid(item_values=[
        DashHeading(title="Explanation", id="explanation-heading"),
        explain_checklist
        ],
                    item_widths=[2, 3], id="explanation_panel")

def create_settings_panel():
    def create_save_settings_grid():
        save_settings_button = DashIconButton(label="Save Settings",
                                              class_name="fa fa-save",
                                              id="save_settings_button")
        return DashGrid(item_values=[
                            save_settings_button,
                            DashDiv(id="saved_to_notification", content=[],
                                    width="100%", visible=False)
                        ],
                        item_widths=[3, 5],
                        id="save_settings_panel")
    settings_app_json_editor = DashJsonEditor(id="settings-app-json-editor")
    settings_data_json_editor = DashJsonEditor(id="settings-data-json-editor")
    save_settings_grid = create_save_settings_grid()
    content = DashGrid(item_values=[
                       settings_app_json_editor,
                       settings_data_json_editor
                       ],
                       item_widths=[None, None],
                       id="settings_panel")
    return DashDiv(id="settings-div",
                   content=[
                       content,
                       save_settings_grid
                   ],
                   width="100%",
                   visible=True)

def create_model_selection_panel(default_selected_dataset):
    data_radio_buttons = DashRadioButton(name="Dataset", options=["train", "valid", "test"],
                                         default_value=default_selected_dataset,
                                         id="data_radiobutton")
    loss_radio_buttons = DashRadioButton(name="Loss", options=get_list_of_supported_standard_metrics(),
                                         default_value="mae",
                                         id="loss_radiobutton")
    return DashGrid(item_values=[DashHeading(title="Automatic Model Selection", id="selection-heading"),
                                 data_radio_buttons,
                                 loss_radio_buttons
                                 ],
                    item_widths=[2, 3, 7], id="auto_selection_panel")

def create_tabs(console, graph,
                settings_panel, model_selection_panel,
                visualization_panel, explanation_panel, gsa_figure_box, error_figure_box):

    def create_button_group(buttons, id):
        return DashButtonGroup(buttons=buttons, id=id)

    data_button_list = [
        DashIconButton(label="Open data folder", class_name="fa fa-folder-open-o", id="select_data_folder_button"),
        DashIconButton(label="Open data settings file", class_name="fa fa-file-o", id="select_data_file_button")
        ]
    data_buttons_group = create_button_group(data_button_list, id="data_button_group")

    data_invisible_buttons_list = [ DashIconButton(label="Load Data", class_name="fa fa-search", id="trigger_load_data_button")]
    data_invisible_buttons_div = DashDiv(id="data_invisible_buttons_div", content=data_invisible_buttons_list, visible=False)

    data_textfield = DashTextarea(id="folder_path_info")
    data_tab_content = [
        data_buttons_group,
        DashNewline(),
        data_textfield,
        data_invisible_buttons_div
    ]

    model_load_button_list = [
        DashIconButton(label="Add models", class_name="fa fa-folder-open-o", id="load_models_button"),
        DashIconButton(label="Add model", class_name="fa fa-file-o", id="load_model_button")
    ]
    model_load_buttons_group = create_button_group(model_load_button_list, id="model_load_button_group")

    model_definition_options = [
        {"label": "XGBoost", "value": "xgb_model_selected"},
        {"label": "Random Forest", "value": "rf_model_selected"},
        {"label": "MLP", "value": "mlp_model_selected"},
        {"label": "ResNet", "value": "resnet_model_selected"},
    ]
    models_definition_dropdown = DashDropdown(options=model_definition_options,
                                              default_value="",
                                              id="models-definition-dropdown")
    models_definition_input = DashInput(id="models-definition-configuration-input",
                  name="",
                  default_value="",
                  width="100%")
    model_definition_stop_training_button = DashIconButton(label="Stop Training",
                                                           class_name="fa fa-stop",
                                                           id="stop_training_button")
    models_definition_button_div = DashDiv(id="models-configuration-button-div",
                                        content=[],
                                        width="100%",
                                        visible=True)
    models_definition_added_models_div = DashDiv(id="models-configuration-added-models-div",
                                        content=[],
                                        width="100%",
                                        visible=True)
    models_definition_stop_training_div = DashDiv(id="models-configuration-stop-training-div",
                                        content=[],
                                        width="100%",
                                        visible=True)

    models_textfield = DashTextarea(id="models_info")
    models_definition_tab_content = [
        DashHeading(title="Load models from file(s): ", id="loadmodels-heading"),
        model_load_buttons_group,
        DashNewline(),
        DashHeading(title="Add model: ", id="addmodel-heading"),
        DashGrid(item_values=[models_definition_dropdown,
                              models_definition_input,
                              model_definition_stop_training_button
                              ],
                 item_widths=[2, 8, 2],
                 id="model_definition_config_grid"),
        DashNewline(),
        DashGrid(item_values=[models_definition_button_div,
                              models_definition_added_models_div,
                              models_definition_stop_training_div
                              ],
                 item_widths=[2, 8, 2],
                 id="model_definition_add_grid"),
        DashNewline(),
        DashHeading(title="Loaded models: ", id="loaded-heading"),
        models_textfield,
        DashNewline(),
        DashGraph(id="progress-bar", figure=None, display_mode_bar=False)
    ]

    model_selection_button_list = [
        DashIconButton(label="Compute error", class_name="fa fa-line-chart", id="compute_errors_button"),
        DashIconButton(label="Search best model", class_name="fa fa-search", id="get_best_model_button")
    ]
    model_selection_buttons_group = DashDiv(id="model_selection_buttons",
                                            content=[DashButtonGroup(buttons=model_selection_button_list, id="model_select_button_group")],
                                            visible=False)
    best_model_texfield = DashTextarea(id="best_model_info")
    model_selection_tab_content = [
        model_selection_buttons_group,
        model_selection_panel,
        DashNewline(),
        best_model_texfield,
        DashNewline(),
        error_figure_box
    ]
    settings_tab_content = [settings_panel]
    console_tab_content = [console]
    plot_tab_content = [visualization_panel,
                        DashNewline(),
                        explanation_panel,
                        DashNewline(),
                        graph
                        ]
    gsa_buttons_list = [ DashIconButton(label="Search most important features", class_name="fa fa-search", id="gsa_best_model_button")]
    gsa_buttons_div = DashDiv(id="gsa_buttons_div", content=gsa_buttons_list, visible=False)
    gsa_texfield = DashTextarea(id="gsa_best_model_info")
    gsa_tab_content = [gsa_buttons_div,
                       DashNewline(),
                       gsa_texfield,
                       DashNewline(),
                       gsa_figure_box
    ]

    tab_list = [DashTab(id="data_tab", label="Data", content=data_tab_content),
                DashTab(id="model_tab", label="Model Definition", content=models_definition_tab_content),
                DashTab(id="evaluation_tab", label="Model Selection", content=model_selection_tab_content),
                DashTab(id="gsa_tab", label="Feature Importance", content=gsa_tab_content),
                DashTab(id="plot_tab", label="Plot", content=plot_tab_content),
                DashTab(id="console_tab", label="Console", content=console_tab_content),
                DashTab(id="settings_tab", label="Settings", content=settings_tab_content)
    ]
    tabs = DashTabGroup(id="tab_group", tabs=tab_list)

    return tabs

def create_main_area(tabs):
    main_area = DashDiv(id="main-div", content=[tabs], margin="0")
    return main_area
