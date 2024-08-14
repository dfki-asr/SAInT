from SAInT.dash_application.dash_component import DashComponent, dcc

class DashTextarea(DashComponent):
    def __init__(self, id, default_value: str = ""):
        super().__init__(id=id)
        self.default_value = default_value
        self.fontsize = "25px"
        self.width = "100%"
        self.height = "550px"

    def to_html(self):
        return dcc.Textarea(id=self.id,
            value=self.default_value,
            readOnly=True,
            style={
                "width": self.width,
                "height": self.height,
                "font-size": self.fontsize
            })
