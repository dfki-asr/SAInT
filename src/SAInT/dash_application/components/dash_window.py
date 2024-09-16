from SAInT.dash_application.dash_component import DashComponent, html

class DashWindow(DashComponent):
    def __init__(self, default_value, id):
        super().__init__(id=id)
        self.default_value = default_value

    def to_html(self, pixel_def):
        return html.Div([
            html.Div(
                id=self.id,
                children=self.default_value,
            )
        ])
