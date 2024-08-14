import io
import matplotlib.pyplot as plt
import re
import base64

class ImageLoader:
    def load_from_file(self, filepath: str):
        with open(filepath, "r") as file:
            data = file.read()
            return data
        return None


    def to_base64(self, data):
        if data is not None:
            if isinstance(data, str):
                data = data.encode()
            data_encoded = base64.b64encode(data).decode()
            return data_encoded
        return None


    def from_base64(self, data_encoded):
        if data_encoded is not None:
            decoded_bytes = base64.b64decode(data_encoded).decode()
            return decoded_bytes
        return None


    def load_html(self, filepath: str):
        data = self.load_from_file(filepath=filepath)
        data_base64 = self.to_base64(data)
        if data_base64 is None:
            return None
        return f"data:text/html;base64,{data_base64}"


    def resize_svg(self, data, width: str, height: str):
        if width and height:
            floating_point_number  = "[-+]?\\d*\\.?\\d+"
            extension = "(pt|px)"
            pattern_str = f"width=\"{floating_point_number}{extension}\" height=\"{floating_point_number}{extension}\" viewBox=\""
            newsize = f"width=\"{width}px\" height=\"{height}px\" viewBox=\""
            pattern = re.compile(pattern_str)
            data = pattern.sub(newsize, data)
        return data


    def load_svg_from_file(self, filepath: str, width: str = None, height: str = None):
        data = self.load_from_file(filepath=filepath)
        data = self.resize_svg(data, width=width, height=height)
        data_base64 = self.to_base64(data)
        if data_base64 is None:
            return None
        return f"data:image/svg+xml;base64,{data_base64}"


    def load_svg_from_plt(self, ax, width: str = None, height: str = None, do_save: bool = False, title: str = "GSA"):
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format="svg")
        if do_save:
            plt.savefig(f"{title}.svg", format="svg")
        image_stream.seek(0)
        data = image_stream.read()
        if not isinstance(data, str):
            data = data.decode()
        data = self.resize_svg(data, width=width, height=height)
        data_base64 = self.to_base64(data)
        return f"data:image/svg+xml;base64,{data_base64}"
