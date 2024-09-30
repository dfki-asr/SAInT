from dash import html, Input, Output
from SAInT.dash_application.feature_importance.dash_global_explainer import DashGlobalExplainer

def register_gsa_callback(dash_app, app):
    @dash_app.callback(
        Output("gsa_best_model_info", "value"),
        Output("gsa_window_div", "children"),
        Input("gsa_best_model_button", "n_clicks"),
        prevent_initial_call=True
    )
    def update_gsa(n_clicks):
        return _handle_gsa_update(app, n_clicks)

def _handle_gsa_update(app, n_clicks):
    if n_clicks:
        _perform_gsa_analysis(app)

    gsa_best_model_info = app.application.most_important_features_text
    gsa_window_div = app.application.gsa_figure
    return gsa_best_model_info, gsa_window_div

def _perform_gsa_analysis(app):
    colors = app.application.color_palette.to_normalized_rgba_list()[:2]
    global_explainer = DashGlobalExplainer(application=app.application)
    gsa_src = global_explainer.explain(method="fast", colors=colors, do_save=True)
    if gsa_src:
        app.application.gsa_figure = [
            html.H6("Global Sensitivity Analysis (GSA)"),
            html.Img(src=gsa_src, width="40%")
        ]
    app.application.model_handler.best_model_was_changed = False
