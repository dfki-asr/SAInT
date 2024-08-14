from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashInput(DashComponent):
    def __init__(self, id, name: str = "", default_value: str = "", width: str = "100px"):
        super().__init__(id=id)
        self.name = name
        self.default_value = default_value
        self.fontsize = "25px"
        self.width = width

    def to_html(self):
        content = []
        if self.name != "":
            content.append(html.H4(self.name))
        content += [dbc.Input(id=self.id,
                      type="text",
                      value=self.default_value,
                      debounce=True,
                      style={
                        "width": self.width,
                        "font-size": self.fontsize
                      }
        )]
        return html.Div(content, style={"font-size": self.fontsize})
