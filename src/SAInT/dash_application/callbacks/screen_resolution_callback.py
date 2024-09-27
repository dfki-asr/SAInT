from dash import html, Input, Output
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
        Input("interval_component", "n_intervals")
    )

    @dash_app.callback(
        Output("screen-dimensions-div", "children"),
        Input("screen-dimensions", "children")
    )
    def update_screen_resolution(screen_dims):
        # If screen dimensions are not available, prevent update
        if screen_dims is None:
            raise PreventUpdate
        # Extract width and height from screen dimensions
        width = screen_dims.get("width")
        height = screen_dims.get("height")
        # Validate screen dimensions
        if width is None or height is None:
            raise PreventUpdate
        # If the dimensions are uninitialized or have changed, update them
        if app.current_screen_dims is None or app.current_screen_dims != screen_dims:
            app.current_screen_dims = screen_dims
            app.application.pixel_definitions = PixelDefinitions(width=width, height=height)
            print(f"Update layout: {width} x {height}")
            return html.Div(f"Screen dimensions: {width} x {height}")
        # If no change in dimensions, prevent update
        raise PreventUpdate
