from dash import Input, Output
from dash.exceptions import PreventUpdate
from SAInT.dash_application.pixel_definitions import PixelDefinitions


def register_screen_resolution_callback(dash_app, app):
    # JavaScript code to get screen dimensions
    dash_app.clientside_callback(
        """
        function(n_intervals) {
            return {
                'width': window.innerWidth,
                'height': window.innerHeight
            };
        }
        """,
        Output("screen-dimensions", "children"),
        Input("interval_component", "n_intervals") # TODO
    )

    @dash_app.callback(
        Output("app_content", "children"),
        Input("screen-dimensions", "children")
    )
    def update_screen_resolution(screen_dims):
        if screen_dims:
            width = screen_dims["width"]
            height = screen_dims["height"]
            # If screen dimensions have changed, update layout
            if app.current_screen_dims != screen_dims:
                app.current_screen_dims = screen_dims
                app.application.pixel_definitions = PixelDefinitions(width=width, height=height)
                app._setup_layout()
                app._register_callbacks()
                layout = app.app.layout
                return layout
            layout = app.app.layout
            return layout
            #raise PreventUpdate
        return "Waiting for screen dimensions..."
