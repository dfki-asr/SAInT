from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashChecklist(DashComponent):
    def __init__(self, name, options, default_value, id, inline: bool = True):
        super().__init__(id=id)
        self.name = name
        self.options = options
        self.default_value = default_value
        self.fontsize = "25px"
        self.inline = inline

    def to_html(self):
        return html.Div([
            html.H4(self.name),
            dbc.Checklist(self.options,
                        self.default_value,
                        inline=self.inline,
                        id=self.id),
        ], style={"font-size": self.fontsize})
