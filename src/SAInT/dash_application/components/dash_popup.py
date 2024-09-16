from SAInT.dash_application.dash_component import DashComponent, dbc, html
from SAInT.dash_application.components import DashIconButton, DashRadioButton

class DashPopup(DashComponent):
    def __init__(self, title: str, id_popup: str, id_window: str, id_close: str, id_save: str = None, fullscreen: bool = True):
        super().__init__(id=id_popup)
        self.title = title
        self.id_popup = id_popup
        self.id_window = id_window
        self.id_close = id_close
        self.id_save = id_save
        self.fullscreen = fullscreen
        self.window = ""
        self.is_open = False

    def reset(self):
        self.window = ""
        self.is_open = False

    def open(self):
        self.is_open = True

    def close(self):
        self.reset()

    def set_content(self, content):
        self.window = dbc.ModalBody(content)

    def get_content(self):
        return self.window

    def to_html(self, pixel_def):
        fontsize = pixel_def.popup_font_size
        close_button = DashIconButton(label="", class_name="fa fa-times", id=self.id_close).to_html(pixel_def)
        title = html.Div([self.title], style={"flex": "1", "textAlign": "left"})
        button = html.Div([close_button], style={"flex": "0", "textAlign": "right"})
        header_content = html.Div([title, button],
                                style={"display": "flex", "alignItems": "center", "width": "100%"})
        content = [
            dbc.ModalHeader(header_content, close_button=False),
            dbc.ModalBody(id=self.id_window)
        ]
        # Footer
        footer = []
        if self.id == "lsa_popup":
            de_normalization_radio_buttons = DashRadioButton(name="Show Data",
                                                    options=["normalized", "denormalized"],
                                                    default_value="denormalized",
                                                    id="norm_denorm_radiobutton")
            footer.append(de_normalization_radio_buttons.to_html(pixel_def))
        if self.id_save:
            save_button = DashIconButton(label="Save", class_name="fa fa-floppy-o", id=self.id_save).to_html(pixel_def)
            footer.append(save_button)
        if len(footer) != 0:
            content.append(
                dbc.ModalFooter(
                    footer
                )
            )
        return dbc.Modal(content,
            id=self.id_popup,
            is_open=self.is_open,
            fullscreen=self.fullscreen,
            keyboard=False,
            backdrop="static",
            style={
                "font-size": fontsize,
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "margin-left": "auto"
            },
            centered=True
        )
