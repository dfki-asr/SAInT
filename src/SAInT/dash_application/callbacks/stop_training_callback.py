from dash import Input, Output
from SAInT.dash_application.common.dash_functions import get_pressed_buttons

def _stop_training(app):
    with open("stop_signal.txt", "w") as f:
        f.write("stop")

def register_stop_training_callback(dash_app, app):
    @dash_app.callback(
        Output("models-configuration-stop-training-div", "children"),
        Input("stop_training_button", "n_clicks")
    )
    def stop_training(n_clicks):
        changed_id = get_pressed_buttons()
        if "stop_training_button.n_clicks" in changed_id:
            if app.application.trainer is not None:
                _stop_training(app)
                return ""
            return ""
        return ""
