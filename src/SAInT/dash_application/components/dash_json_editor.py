from SAInT.dash_application.dash_component import DashComponent, html, dcc
from SAInT.dash_application.pixel_definitions import text_font_size, editor_window_height

class DashJsonEditor(DashComponent):
    def __init__(self, id, default_value: str = ""):
        super().__init__(id=id)
        self.default_value = default_value
        self.fontsize = text_font_size
        self.width = "100%"
        self.height = editor_window_height

    def to_html(self):
        return html.Div([
            html.H4(f"{self.id}"),
            dcc.Textarea(
            id=self.id,
            style={"width": self.width,
                "height": self.height,
                "font-size": self.fontsize}
            )]
        )
