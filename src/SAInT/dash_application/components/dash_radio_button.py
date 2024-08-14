from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashRadioButton(DashComponent):
    def __init__(self, name, options, default_value, id):
        super().__init__(id=id)
        self.name = name
        self.options = options
        self.default_value = default_value
        self.fontsize = "25px"

    def to_html(self):
        return html.Div([
            html.H4(self.name),
            dbc.RadioItems(self.options,
                       self.default_value,
                       inline=True,
                       id=self.id),
        ], style={"font-size": self.fontsize})
