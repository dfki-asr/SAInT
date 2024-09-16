from SAInT.dash_application.dash_component import DashComponent, html

class DashDiv(DashComponent):
    def __init__(self, id,
                 content,
                 width: str = None,
                 margin: str = None,
                 visible: bool = True,
                 inline: bool = False,
                 gap: str = None):
        super().__init__(id=id)
        self.content = content
        self.width = width
        self.margin = margin
        self.visible = visible
        self.inline = inline
        self.gap = gap

    def to_html(self, pixel_def):
        fontsize = pixel_def.text_font_size
        content = [item.to_html(pixel_def) for item in self.content]
        style = {}
        if self.width is not None:
            style["width"] = self.width
        if self.margin is not None:
            style["margin"] = self.margin
        if fontsize is not None:
            style["font-size"] = fontsize
        if self.visible is False:
            style["display"] = "none"
        elif self.inline:  # Apply flexbox if inline is True
            style["display"] = "flex"
            style["flex-direction"] = "row"
            if self.gap is not None:  # Add gap if specified
                style["gap"] = self.gap
        else:
            style["display"] = "block"  # Default to block display
        style = style if len(style) > 0 else None
        return html.Div(
            content,
            id=self.id,
            style=style,
        )
