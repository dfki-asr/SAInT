from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashDropdown(DashComponent):
    def __init__(self, options, default_value, id):
        super().__init__(id=id)
        self.options = options
        self.default_value = default_value
        self.fontsize = "25px"

    def to_html(self):
        return html.Div([
            dbc.Select(
                id=self.id,
                options=self.options,
                value=self.default_value,
                style={"width": "300px",
                    "font-size": self.fontsize}
            ),
        html.Div(id=f"{self.id}-output")
        ])
