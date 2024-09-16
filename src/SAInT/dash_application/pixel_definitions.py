class PixelDefinitions:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.text_font_size = "12px"
        self.popup_font_size = "14px"
        self.border_line = "0.5px solid #000"
        self.border_radius = "7px"
        self.margin = "15px"
        self.padding = "7px"
        self.logo_height = "6.5%"
        self.lime_expl_width = "65%"
        self.tab_gap = "1.443%"

    def calc_rel_size(self, ref_value, percentage, unit):
        result = int(ref_value / 100 * percentage)
        if unit == "px":
            return f"{result}px"
        else:
            return result

    @property
    def title_font_size(self):
        return self.calc_rel_size(self.width, 1.985, "px")

    @property
    def feature_popup_width(self):
        return self.calc_rel_size(self.width, 30, "px")

    @property
    def error_plot_width(self):
        return self.calc_rel_size(self.width, 50, "px")

    @property
    def figure_width(self):
        return self.calc_rel_size(self.width, 93, "-")

    @property
    def marker_size(self):
        return self.calc_rel_size(self.width, 0.8, "-")

    @property
    def area_height(self):
        return self.calc_rel_size(self.height, 25, "px")

    @property
    def editor_window_height(self):
        return self.calc_rel_size(self.height, 70, "px")

    @property
    def lime_height(self):
        return self.calc_rel_size(self.height, 32, "px")

    @property
    def shap_height(self):
        return self.calc_rel_size(self.height, 22, "px")

    @property
    def error_plot_height(self):
        return self.calc_rel_size(self.height, 15, "px")

    @property
    def default_figure_height(self):
        return self.calc_rel_size(self.height, 62, "-")
