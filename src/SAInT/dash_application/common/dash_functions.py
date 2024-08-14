from dash import callback_context

def get_pressed_buttons():
    changed_id = [p["prop_id"] for p in callback_context.triggered][0]
    return changed_id
