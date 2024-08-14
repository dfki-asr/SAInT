from dash import Input, Output

def _handle_update_progress_bar(app):
    current_std_output = app.console.stdout_buffer.getvalue()
    return app.progress_bar.generate_progress_bar(current_std_output)

def register_training_process_bar_callback(dash_app, app):
    @dash_app.callback(
        Output("progress-bar", "figure"),
        Input("interval_component", "n_intervals")
    )
    def update_progress_bar(n):
        return _handle_update_progress_bar(app)
