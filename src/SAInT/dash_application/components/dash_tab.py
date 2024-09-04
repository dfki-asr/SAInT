from SAInT.dash_application.dash_component import DashComponent, dbc
from SAInT.dash_application.pixel_definitions import popup_font_size, tab_gap, padding

class DashTab(DashComponent):
    def __init__(self, id, label, content):
        super().__init__(id=id)
        self.label = label
        self.content = content
        self.fontsize = popup_font_size

    def to_html(self):
        tab_content = dbc.Card(dbc.CardBody(
            [item.to_html() for item in self.content],
            style={
                "padding": padding
            })
        )
        return dbc.Tab(
            children=tab_content,
            label=self.label,
            label_style={"padding": padding},
            tab_style={
                "font-size": self.fontsize,
                "margin-right": tab_gap
            }
        )
