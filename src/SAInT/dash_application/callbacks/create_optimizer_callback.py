from dash import Input, Output

def register_create_optimizer_callback(dash_app, app):
    @dash_app.callback(Output("create_optimizer_button", "n_clicks"),
                       Input("gsa_best_model_button", "n_clicks"),
                       prevent_initial_call=True)
    def auto_create_optimizer(n_clicks):
        return _handle_optimizer_update(app, n_clicks)

def register_optimize_callback(dash_app, app):
    @dash_app.callback(
        Output("optimizer_div", "children"),
        Input("create_optimizer_button", "n_clicks"),
        prevent_initial_call=True
    )
    def update_optimization(n_clicks):
        if n_clicks:
            return _handle_optimizer_update(app, n_clicks)

def _handle_optimizer_update(app, n_clicks):
    if n_clicks:
        print("Create Optimizer Example")
        div_content = _create_optimizer_example(app)
        # ToDo: How to set this correctly?
        optimizer_div = div_content
    else:
        optimizer_div = ""
    return optimizer_div

def _create_optimizer_example(app):
    print("CREATE OPT")
    div_content = app.application.optimizer.prepare()
    print(div_content)
    #colors = app.application.color_palette.to_normalized_rgba_list()[:2]
    #global_explainer = DashGlobalExplainer(application=app.application)
    #app.application.model_handler.best_model_was_changed = False
    return div_content
