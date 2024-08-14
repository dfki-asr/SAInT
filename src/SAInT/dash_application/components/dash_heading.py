
from SAInT.dash_application.dash_component import DashComponent, html

class DashHeading(DashComponent):
    def __init__(self, title, id: str = None):
        super().__init__(id=id)
        self.title = title

    def to_html(self):
        return html.H4(self.title)
