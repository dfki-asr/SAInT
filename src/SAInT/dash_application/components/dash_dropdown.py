from SAInT.dash_application.dash_component import DashComponent, html, dbc
from SAInT.dash_application.pixel_definitions import text_font_size

class DashDropdown(DashComponent):
    def __init__(self, options, default_value, id):
        super().__init__(id=id)
        self.options = options
        self.default_value = default_value
        self.fontsize = text_font_size

    def to_html(self):
        return html.Div([
            dbc.Select(
                id=self.id,
                options=self.options,
                value=self.default_value,
                style={"width": "100%",
                    "font-size": self.fontsize}
            ),
        html.Div(id=f"{self.id}-output")
        ])
