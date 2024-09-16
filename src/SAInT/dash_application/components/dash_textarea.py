from SAInT.dash_application.dash_component import DashComponent, dcc

class DashTextarea(DashComponent):
    def __init__(self, id, default_value: str = ""):
        super().__init__(id=id)
        self.default_value = default_value
        self.width = "100%"

    def to_html(self, pixel_def):
        fontsize = pixel_def.text_font_size
        height = pixel_def.area_height
        return dcc.Textarea(id=self.id,
            value=self.default_value,
            readOnly=True,
            style={
                "width": self.width,
                "height": height,
                "font-size": fontsize
            })
