from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashButton(DashComponent):
    def __init__(self, content, id):
        super().__init__(id=id)
        self.content = content
        self.fontsize = "25px"
        self.border_line = "1px solid #000"
        self.border_radius = "10px"

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
                   "border-radius": self.border_radius},
            n_clicks=0)
