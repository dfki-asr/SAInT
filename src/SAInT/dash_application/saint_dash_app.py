#!/usr/bin/env python

import asyncio
import webbrowser
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from SAInT.dash_application.saint_application import SAInTApplication
from SAInT.dash_application import saint_dash_layout_components as layout
from SAInT.dash_application.data_definition.data_handler import DataHandler
from SAInT.dash_application.model_definition.progress_bar import ProgressBar
from SAInT.dash_application.common.image_loader import ImageLoader
from SAInT.dash_application.console.console import Console
from SAInT.dash_application.callbacks import *
from SAInT.dash_application.components.dash_interval import DashInterval
import importlib.resources as pkg_resources
from SAInT.dash_application import __name__ as dash_app_module

class MySAInTDashApplication:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[
            dbc.themes.LUX,
            "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
        ])
        self.application = SAInTApplication()
        self.console = Console()
        self.data_handler = DataHandler(application=self.application)
        self.progress_bar = ProgressBar()
        # Store current screen dimensions to check for changes
        self.current_screen_dims = None
        self.setup_done = False
        self.registered_callbacks_done = False
        self.interval_component = DashInterval(interval_in_ms=500.0, id="interval_component")
        # Initial basic layout to include the necessary components
        self._setup_layout()
        self._register_callbacks()

    def _setup_layout(self):
        pixel_def = self.application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        title = "SAInT: An Interactive Tool for Sensitivity Analysis In The Loop"
        figure = self.application.interactive_plot.figure
        image_loader = ImageLoader()
        logo_src = ""
        with pkg_resources.path(dash_app_module, "logo.svg") as logo_svg_filepath:
            logo_src = image_loader.load_svg_from_file(filepath=str(logo_svg_filepath))
        graph = layout.create_graph(figure=figure, id="graph")
        console = layout.create_console()
        gsa_figure_box = layout.create_gsa_window()
        error_figure_box = layout.create_error_window()
        header = layout.create_header(title=title, logo=logo_src)

        interactive_plot = self.application.interactive_plot
        default_selected_dataset = interactive_plot.dataset_selection
        model_selection_panel = layout.create_model_selection_panel(default_selected_dataset)
        default_sort_criterion = interactive_plot.sort_criterion
        default_goodness_of_fit = str(interactive_plot.goodness_of_fit)
        default_show_feature_details = str(interactive_plot.show_feature_details)
        visualization_panel = layout.create_visualization_panel(
            default_selected_dataset, default_sort_criterion,
            default_goodness_of_fit,
            default_show_feature_details
        )
        explanation_panel = layout.create_explanation_panel()
        settings_panel = layout.create_settings_panel()

        tabs = layout.create_tabs(
            console=console,
            graph=graph,
            settings_panel=settings_panel,
            model_selection_panel=model_selection_panel,
            visualization_panel=visualization_panel,
            explanation_panel=explanation_panel,
            gsa_figure_box=gsa_figure_box,
            error_figure_box=error_figure_box
        )

        invisible_divs = [
            layout.create_invisible_div(id="update_panel_from_settings"),
            layout.create_invisible_div(id="loaded_data_settings"),
            layout.create_invisible_div(id="trigger_load_data_after_feature_selection"),
            layout.create_invisible_div(id="loaded_data"),
            layout.create_invisible_div(id="models_trained_info"),
            layout.create_invisible_div(id="models_loaded_info"),
            layout.create_invisible_div(id="settings-app-json-editor-div"),
            layout.create_invisible_div(id="settings-data-json-editor-div"),
            layout.create_invisible_div(id="screen-dimensions-div"),
            layout.create_invisible_div(id="scan-bias-div"),
            layout.create_invisible_div(id="create-lsa-explainers-div")
        ]

        div_content = [header,
                       layout.create_main_area(tabs=tabs),
                       self.application.lsa_popup,
                       self.application.feature_selection_popup,
                       *invisible_divs]
        pixel_def = self.application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        self.app.layout = html.Div([
            layout.create_link(
                rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
            ).to_html(pixel_def),
            html.Div(id="screen-dimensions", style={"display": "none"}),
            self.interval_component.to_html(pixel_def),
            dcc.ConfirmDialog(id="data-error-dialog", message=""),
            dcc.ConfirmDialog(id="feature-error-dialog", message=""),
            layout.create_div(id="main-content-div", content=div_content, margin="1%").to_html(pixel_def)
        ], id="app_content")
        self.setup_done = True

    def _register_callbacks(self):
        if self.registered_callbacks_done:
            return
        callbacks_to_register = [
            register_data_callback,
            register_model_loading_callback,
            register_model_loaded_or_trained_callback,
            register_plot_callback,
            register_gsa_callback,
            register_console_callback,
            register_lsa_popup_callback,
            register_open_feature_selection_popup_callback,
            register_close_feature_selection_popup_callback,
            register_training_process_bar_callback,
            register_load_settings_callback,
            register_save_settings_callback,
            register_update_local_explain_callback,
            register_data_settings_callback,
            register_auto_compute_error_callback,
            register_auto_gsa_callback,
            register_model_configuration_callback,
            register_add_model_callback,
            register_stop_training_callback,
            register_update_app_settings_callback,
            register_update_data_settings_callback,
            register_screen_resolution_callback,
            register_auto_reloaddata_callback,
            register_create_local_explainers_callback
        ]
        for callback_func in callbacks_to_register:
            callback_func(self.app, self)
        self.registered_callbacks_done = True

    def run(self):
        async def start_server():
            await self.app.run_server(debug=False)

        async def open_browser():
            await webbrowser.open("http://127.0.0.1:8050/")

        loop = asyncio.get_event_loop()
        browser_task = loop.create_task(open_browser())
        server_task = loop.create_task(start_server())
        if not loop.is_running():
            loop.run_until_complete(asyncio.gather(browser_task, server_task))
