from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashButton(DashComponent):
    def __init__(self, content, id):
        super().__init__(id=id)
        self.content = content

    def to_html(self, pixel_def):
        fontsize = pixel_def.text_font_size
        border_line = pixel_def.border_line
        border_radius = pixel_def.border_radius
        padding = pixel_def.padding
        content = self.content
        if len(content) > 1:
            class_name, label = content
            content = [html.I(className=class_name), label]
        return dbc.Button(content,
            id=self.id,
            color="secondary",
            style={"font-size": fontsize,
                   "border": border_line,
                   "border-radius": border_radius,
                   "line-height": "1.2",
                   "padding": padding
                   },
            n_clicks=0)
