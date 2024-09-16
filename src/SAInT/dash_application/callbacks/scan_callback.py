from dash import html, Input, Output

def register_scan_bias_callback(dash_app, app):
    @dash_app.callback(
        Output("scan-bias-div", "children"),
        Input("gsa_best_model_button", "n_clicks"),
        prevent_initial_call=True
    )
    def update_scan_bias(n_clicks):
        return _handle_scan_bias_update(app, n_clicks)

def _handle_scan_bias_update(app, n_clicks):
    if n_clicks:
        _perform_scan_for_bias(app)

def _perform_scan_for_bias(app):
    if app.application.trainer:
        app.application.local_explainer.scan_and_filter_samples()
    return ""
