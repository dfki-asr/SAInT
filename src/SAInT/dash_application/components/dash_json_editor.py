from SAInT.dash_application.dash_component import DashComponent, html, dcc

class DashJsonEditor(DashComponent):
    def __init__(self, id, default_value: str = ""):
        super().__init__(id=id)
        self.default_value = default_value
        self.fontsize = "25px"
        self.width = "100%"
        self.height = "1300px"

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
