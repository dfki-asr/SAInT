from SAInT.dash_application.dash_component import DashComponent, html, dbc

class DashGrid(DashComponent):
    def __init__(self, item_values, item_widths, id):
        super().__init__(id=id)
        self.item_values = item_values
        self.item_widths = item_widths

    def to_html(self, pixel_def):
        cols = [
            dbc.Col(item.to_html(pixel_def), width=width) if width is not None else dbc.Col(item.to_html(pixel_def)) for item, width in zip(self.item_values, self.item_widths)
        ]
        content = [dbc.Row(cols)]
        return html.Div(content)
