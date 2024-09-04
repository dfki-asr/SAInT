from SAInT.dash_application.dash_component import DashComponent, html, dbc
from SAInT.dash_application.pixel_definitions import text_font_size, border_line, border_radius, padding

class DashButton(DashComponent):
    def __init__(self, content, id):
        super().__init__(id=id)
        self.content = content
        self.fontsize = text_font_size
        self.border_line = border_line
        self.border_radius = border_radius

    def to_html(self):
        content = self.content
        if len(content) > 1:
            class_name, label = content
            content = [html.I(className=class_name), label]
        return dbc.Button(content,
            id=self.id,
            color="secondary",
            style={"font-size": self.fontsize,
                   "border": self.border_line,
                   "border-radius": self.border_radius,
                   "padding": padding
                   },
            n_clicks=0)
