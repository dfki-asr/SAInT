
from SAInT.dash_application.dash_component import DashComponent, html
from SAInT.dash_application.pixel_definitions import title_font_size, logo_height, margin

class DashHeader(DashComponent):
    def __init__(self, title, logo):
        super().__init__(id="header")
        self.title = title
        self.logo = logo
        self.fontsize = title_font_size
        self.margin_right = margin

    def to_html(self):
        return html.Div([
            # Container for the title
            html.H1(
                self.title,
                style={
                    "flex-grow": "1",
                    "font-size": self.fontsize,
                    "text-align": "center",
                    "margin": "0"
                }
            ),
            # Container for the logo
            html.Img(
                src=self.logo,
                style={
                    "height": logo_height,
                    "margin-left": "auto",
                    "margin-right": self.margin_right
                }
            )
        ], style={
            "display": "flex",
            "flex-direction": "row",
            "align-items": "center",
            "width": "100%"
        })
