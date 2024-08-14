from SAInT.dash_application.dash_component import DashComponent, dbc

class DashTab(DashComponent):
    def __init__(self, id, label, content):
        super().__init__(id=id)
        self.label = label
        self.content = content
        self.fontsize = "30px"

    def to_html(self):
        tab_content = dbc.Card(dbc.CardBody([item.to_html() for item in self.content]))
        return dbc.Tab(
            children=tab_content,
            label=self.label,
            tab_style={'font-size': self.fontsize}
        )
