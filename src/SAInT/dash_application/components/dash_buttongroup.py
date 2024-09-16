from SAInT.dash_application.dash_component import DashComponent, html

class DashButtonGroup(DashComponent):
    def __init__(self, buttons, id):
        super().__init__(id=id)
        self.buttons = buttons

    def to_html(self, pixel_def):
        margin = pixel_def.margin
        content = []
        num_buttons = len(self.buttons)
        for button_idx, button in enumerate(self.buttons):
            if button_idx == num_buttons-1:
                content.append(html.Div([button.to_html(pixel_def)],
                                        style={"align-self": "flex-end"}))
            else:
                content.append(html.Div([button.to_html(pixel_def)],
                                        style={"margin-right": margin}))
        return html.Div(
            content,
            style={
            "display": "flex",
            "flex-direction": "row"
        })
