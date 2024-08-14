from SAInT.dash_application.dash_component import DashComponent, dbc

class DashTabGroup(DashComponent):
    def __init__(self, id, tabs):
        super().__init__(id=id)
        self.tabs = tabs

    def to_html(self):
        content = [tab.to_html() for tab in self.tabs]
        return dbc.Tabs(content)
