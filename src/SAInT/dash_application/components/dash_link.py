
from SAInT.dash_application.dash_component import DashComponent, html

class DashLink(DashComponent):
    def __init__(self,
                 rel: str = "stylesheet",
                 href: str = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"):
        super().__init__(id=id)
        self.rel = rel
        self.href = href

    def to_html(self, pixel_def):
        return html.Link(
            rel=self.rel,
            href=self.href
        )
