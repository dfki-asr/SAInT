from SAInT.dash_application.dash_component import DashComponent, dcc

class DashGraph(DashComponent):
    def __init__(self, figure, id, width: str = "100%", display_mode_bar: bool = True):
        super().__init__(id=id)
        self.figure = figure
        self.width = width
        self.display_mode_bar = display_mode_bar

    def to_html(self, pixel_def):
        return dcc.Graph(
            figure=self.figure,
            id=self.id,
            style={"width": self.width},
            config={"displayModeBar": self.display_mode_bar}
        )
