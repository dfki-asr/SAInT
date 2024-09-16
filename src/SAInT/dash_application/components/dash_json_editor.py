from SAInT.dash_application.dash_component import DashComponent, html, dcc

class DashJsonEditor(DashComponent):
    def __init__(self, id, default_value: str = ""):
        super().__init__(id=id)
        self.default_value = default_value
        self.width = "100%"

    def to_html(self, pixel_def):
        fontsize = pixel_def.text_font_size
        height = pixel_def.editor_window_height
        return html.Div([
            html.H6(f"{self.id}"),
            dcc.Textarea(
            id=self.id,
            style={"width": self.width,
                "height": height,
                "font-size": fontsize}
            )]
        )
