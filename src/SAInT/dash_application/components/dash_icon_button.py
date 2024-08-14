from SAInT.dash_application.components.dash_button import DashButton

class DashIconButton(DashButton):
    def __init__(self, label, class_name, id):
        if label != "":
            label = "     " + label
        content = [class_name, label]
        super().__init__(content, id)
