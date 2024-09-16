
from SAInT.dash_application.dash_component import DashComponent, html

class DashHeader(DashComponent):
    def __init__(self, title, logo):
        super().__init__(id="header")
        self.title = title
        self.logo = logo

    def to_html(self, pixel_def):
        fontsize = pixel_def.title_font_size
        logo_height = pixel_def.logo_height
        margin_right = pixel_def.margin
        return html.Div([
            # Container for the title
            html.H1(
                self.title,
                style={
                    "flex-grow": "1",
                    "font-size": fontsize,
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
                    "margin-right": margin_right
                }
            )
        ], style={
            "display": "flex",
            "flex-direction": "row",
            "align-items": "center",
            "width": "100%"
        })
