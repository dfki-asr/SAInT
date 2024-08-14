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
