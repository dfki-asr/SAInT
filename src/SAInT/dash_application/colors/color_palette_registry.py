from SAInT.dash_application.colors.color_palette import ColorPalette
from SAInT.dash_application.colors.color import Color

class ColorPaletteRegistry:
    def __init__(self):
        """
        Initialize a ColorPaletteRegistry instance.
        """
        self.palette_dict = {}

    def setup(self):
        """
        Setup default palettes in the registry.
        """
        self.register_palette(ColorPalette("dfki", [
            Color("dfki_pink", 236, 97, 159),
            Color("dfki_green", 106, 191, 163),
            Color("dfki_blue", 29, 58, 143),
            Color("dfki_orange", 247, 167, 18),
            Color("dfki_dark", 6, 23, 28),
            Color("dfki_gray", 215, 219, 221),
            Color("jet", 47, 45, 46)
        ]))

        self.register_palette(ColorPalette("retro", [
            Color("carrot_orange", 241, 143, 1),
            Color("blue_munsell", 4, 139, 168),
            Color("charcoal", 46, 64, 87),
            Color("yellow_green", 153, 194, 77),
            Color("wine", 99, 43, 48),
            Color("naples_yellow", 250, 223, 99),
            Color("jet", 47, 45, 46)
        ]))

        self.register_palette(ColorPalette("warm", [
            Color("misty_rose", 241, 223, 218),
            Color("rosy_brown", 211, 150, 146),
            Color("jasper", 194, 96, 81),
            Color("burnt_sienna", 212, 120, 84),
            Color("wine", 99, 43, 48),
            Color("champagne_pink", 241, 214, 192),
            Color("jet", 47, 45, 46),
        ]))

        self.register_palette(ColorPalette("colorful", [
            Color("pear", 194, 232, 18),
            Color("royal_purple", 116, 82, 150),
            Color("light_green", 145, 245, 173),
            Color("cadet_gray", 139, 158, 183),
            Color("light_coral", 231, 143, 142),
            Color("violet", 99, 42, 80),
            Color("black_olive", 72, 69, 56)
        ]))

        self.register_palette(ColorPalette("saint", [
            Color("blue_munsell", 30, 156, 172),
            Color("yellow_green", 153, 194, 77),
            Color("gray", 127, 127, 127),
            Color("carrot_orange", 241, 143, 1),
            Color("apricot", 251, 196, 171),
            Color("davys_gray", 77, 77, 77),
            Color("black_olive", 72, 69, 56)
        ]))

    def register_palette(self, palette: ColorPalette):
        """
        Register a color palette in the registry.
        :param palette: ColorPalette instance to register.
        """
        self.palette_dict[palette.name] = palette
