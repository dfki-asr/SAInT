from SAInT.dash_application.dash_component import DashComponent, dbc

class DashTab(DashComponent):
    def __init__(self, id, label, content):
        super().__init__(id=id)
        self.label = label
        self.content = content

    def to_html(self, pixel_def):
        fontsize = pixel_def.popup_font_size
        tab_gap = pixel_def.tab_gap
        padding = pixel_def.padding
        tab_content = dbc.Card(dbc.CardBody(
            [item.to_html(pixel_def) for item in self.content],
            style={
                "padding": padding
            })
        )
        return dbc.Tab(
            children=tab_content,
            label=self.label,
            label_style={"padding": padding},
            tab_style={
                "font-size": fontsize,
                "margin-right": tab_gap
            }
        )
