
from SAInT.dash_application.dash_component import DashComponent, html

class DashNewline(DashComponent):
    def __init__(self, id: str = None):
        super().__init__(id=id)

    def to_html(self, pixel_def):
        return html.Br()