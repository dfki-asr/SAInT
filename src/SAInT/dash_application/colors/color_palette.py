class ColorPalette:
    def __init__(self, name: str, colors: list):
        """
        Initialize a ColorPalette instance.
        :param name: Name of the palette.
        :param colors: List of Color objects.
        """
        self.name = name
        self.colors = colors

    def to_rgba_dict(self) -> dict:
        """
        Return a dictionary of color names to RGBA strings.
        :return: Dictionary of color names to RGBA strings.
        """
        return {color.name: color.to_rgba_str() for color in self.colors}

    def to_normalized_rgba_dict(self) -> dict:
        """
        Return a dictionary of color names to normalized RGBA tuples.
        :return: Dictionary of color names to normalized RGBA tuples.
        """
        return {color.name: color.to_normalized_rgba_tuple() for color in self.colors}

    def to_decimal_list(self) -> list:
        """
        Return a list of RGBA strings for the colors in the palette.
        :return: List of decimal strings.
        """
        return [color.to_decimal() for color in self.colors]

    def to_hex_list(self) -> list:
        """
        Return a list of RGBA strings for the colors in the palette.
        :return: List of Hex strings.
        """
        return [color.to_hex_str() for color in self.colors]

    def to_rgba_list(self) -> list:
        """
        Return a list of RGBA strings for the colors in the palette.
        :return: List of RGBA strings.
        """
        return [color.to_rgba_str() for color in self.colors]

    def to_normalized_rgba_list(self) -> list:
        """
        Return a list of normalized RGBA tuples for the colors in the palette.
        :return: List of normalized RGBA tuples.
        """
        return [color.to_normalized_rgba_tuple() for color in self.colors]
