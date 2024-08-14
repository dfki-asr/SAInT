class Color:
    def __init__(self, name: str, r: int, g: int, b: int, a: int = 255):
        """
        Initialize a Color instance.
        :param name: Name of the color.
        :param r: Red component (0-255).
        :param g: Green component (0-255).
        :param b: Blue component (0-255).
        :param a: Alpha component (0-255).
        """
        self.name = name
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    @classmethod
    def from_decimal(cls, encoded_value):
        if isinstance(encoded_value, str):
            encoded_value = int(encoded_value)
        r = (encoded_value >> 16) & 0xFF
        g = (encoded_value >> 8) & 0xFF
        b = encoded_value & 0xFF
        return cls(r, g, b)

    def to_decimal(self) -> int:
        decimal = (self.r << 16) | (self.g << 8) | (self.b)
        return decimal

    def to_hex_str(self):
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}{self.a:02x}"

    def to_rgba_str(self) -> str:
        """
        Return the color as an RGBA string.
        :return: RGBA string.
        """
        return f"rgba({self.r}, {self.g}, {self.b}, {self.a})"

    def to_normalized_rgba_tuple(self) -> tuple:
        """
        Return the color as a tuple of normalized RGBA values.
        :return: Tuple of (r, g, b, a) with normalized RGBA values.
        """
        return (self.r / 255, self.g / 255, self.b / 255, self.a / 255)
