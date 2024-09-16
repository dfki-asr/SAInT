from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashRadioButton(DashComponent):
    def __init__(self, name, options, default_value, id):
        super().__init__(id=id)
        self.name = name
        self.options = options
        self.default_value = default_value

    def to_html(self, pixel_def):
        fontsize = pixel_def.text_font_size
        return html.Div([
            html.H6(self.name),
            dbc.RadioItems(self.options,
                       self.default_value,
                       inline=True,
                       id=self.id),
        ], style={"font-size": fontsize})
