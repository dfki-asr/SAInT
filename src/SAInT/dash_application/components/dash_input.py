from SAInT.dash_application.dash_component import DashComponent, html, dbc
from SAInT.dash_application.pixel_definitions import text_font_size, default_input_width

class DashInput(DashComponent):
    def __init__(self, id, name: str = "", default_value: str = "", width: str = default_input_width):
        super().__init__(id=id)
        self.name = name
        self.default_value = default_value
        self.fontsize = text_font_size
        self.width = width

    def to_html(self):
        content = []
        if self.name != "":
            content.append(html.H6(self.name))
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
