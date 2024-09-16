from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashDropdown(DashComponent):
    def __init__(self, options, default_value, id):
        super().__init__(id=id)
        self.options = options
        self.default_value = default_value

    def to_html(self, pixel_def):
        fontsize = pixel_def.text_font_size
        return html.Div([
            dbc.Select(
                id=self.id,
                options=self.options,
                value=self.default_value,
                style={"width": "100%",
                    "font-size": fontsize}
            ),
        html.Div(id=f"{self.id}-output")
        ])
