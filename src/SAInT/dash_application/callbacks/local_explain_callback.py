from dash import Input, Output

def _handle_update_local_explain(app, explain_checklist):
    if app.application.settings is not None:
        app.application.local_explainer.explain_lime = "lime" in explain_checklist
        app.application.local_explainer.explain_shap = "shap" in explain_checklist
    return explain_checklist

def register_update_local_explain_callback(dash_app, app):
    @dash_app.callback(
        Output("explain_checklist", "value"),
        Input("explain_checklist", "value")
    )
    def update_local_explain(explain_checklist):
        return _handle_update_local_explain(app, explain_checklist)

def register_create_local_explainers_callback(dash_app, app):
    @dash_app.callback(
        Output("create-lsa-explainers-div", "children"),
        Input("gsa_best_model_button", "n_clicks"),
        prevent_initial_call=True
    )
    def create_local_explainers(n_clicks):
        if n_clicks:
            if app.application.trainer:
                app.application.local_explainer.create_explainer_instances()
        return ""
