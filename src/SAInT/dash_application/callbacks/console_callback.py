from dash import Input, Output

def register_console_callback(dash_app, app):
    @dash_app.callback(
        Output("output_textarea", "value"),
        Input("interval_component", "n_intervals"),
        Input("clear_console_button", "n_clicks")
    )
    def update_console(n_intervals, n_clicks):
        return _handle_console_update(app, n_intervals, n_clicks)

def _handle_console_update(app, n_intervals, n_clicks):
    return app.console.update()
