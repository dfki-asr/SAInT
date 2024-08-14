from SAInT.dash_application.dash_component import DashComponent, html

class DashDiv(DashComponent):
    def __init__(self, id,
                 content,
                 width: str = None,
                 margin: str = None,
                 visible: bool = True):
        super().__init__(id=id)
        self.content = content
        self.fontsize = "25px"
        self.width = width
        self.margin = margin
        self.visible = visible

    def to_html(self):
        content = [item.to_html() for item in self.content]
        style = {}
        if self.width is not None:
            style["width"] = self.width
        if self.margin is not None:
            style["margin"] = self.margin
        if self.fontsize is not None:
            style["font-size"] = self.fontsize
        if self.visible is False:
            style["display"] = "none"
        style = style if len(style) > 0 else None
        return html.Div(
            content,
            id=self.id,
            style=style,
        )
