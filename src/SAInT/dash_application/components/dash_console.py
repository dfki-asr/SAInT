from SAInT.dash_application.dash_component import DashComponent, html, dcc
from SAInT.dash_application.components.dash_icon_button import DashIconButton
from SAInT.dash_application.components.dash_interval import DashInterval

class DashConsole(DashComponent):
    def __init__(self, interval_in_ms: float = 500.0):
        super().__init__(id="output_textarea")
        self.name = "Console"
        self.default_value = ""
        self.fontsize = "25px"
        self.width = "100%"
        self.height = "550px"
        self.clear_button = DashIconButton(
            label="Clear console",
            class_name="fa fa-times",
            id="clear_console_button")
        self.interval_component = DashInterval(
            interval_in_ms=interval_in_ms,
            id="interval_component")

    def to_html(self):
        return html.Div([
        html.H3(self.name),
        dcc.Textarea(id=self.id,
                    value=self.default_value,
                    readOnly=True,
                    style={
                        "width": self.width,
                        "height": self.height,
                        "font-size": self.fontsize
                    }),
        self.interval_component.to_html(),
        self.clear_button.to_html()
    ])
