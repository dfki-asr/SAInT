import re
from plotly.graph_objects import Figure

class ProgressBar:
    def __init__(self):
        self.pattern = self._compile_epoch_pattern()

    @staticmethod
    def _compile_epoch_pattern():
        """Compile regex pattern to find epoch progress strings."""
        pattern_str = r"Epoch \d+/\d+"
        return re.compile(pattern_str)

    @staticmethod
    def _calculate_progress_percentage(matches):
        """Calculate progress percentage from matched epoch strings."""
        if not matches:
            return 0
        last_match = matches[-1] if isinstance(matches, list) else matches
        epochs, max_epochs = map(int, last_match.replace("Epoch ", "").split("/"))
        return int(((epochs - 1) * 100.0) / max_epochs)

    @staticmethod
    def _find_matches(pattern, text):
        """Find all matches of the pattern in the given text."""
        return re.findall(pattern, text)

    @staticmethod
    def _add_rectangle_to_figure(figure, color):
        """Draw a rectangle border on the given figure."""
        lines = [
            ((0, 0), (100, 0)),
            ((0, 1), (100, 1)),
            ((0, 0), (0, 1)),
            ((100, 0), (100, 1))
        ]
        for start, end in lines:
            figure.add_shape(
                type='line',
                x0=start[0], y0=start[1],
                x1=end[0], y1=end[1],
                line=dict(color=color, width=1)
            )
        return figure

    @staticmethod
    def _create_default_figure():
        """Create a default empty figure."""
        figure = Figure(data=[])
        figure.update_layout(
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        return figure

    def _create_progress_bar_figure(self, percent):
        """Create a figure showing the progress bar at the given percentage."""
        progress = percent % 101
        marker_color = "black"
        progress_bar_data = [{
            "x": [progress],
            "y": [0.5],
            "type": "bar",
            "orientation": "h",
            "marker": {"color": marker_color, "line": {"width": 1}}
        }]
        figure = Figure(progress_bar_data)
        figure.update_layout(
            height=200,
            font=dict(size=20),
            showlegend=False,
            xaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(size=20)),
            yaxis=dict(showticklabels=False, tickfont=dict(size=20)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return self._add_rectangle_to_figure(figure, marker_color)

    def generate_progress_bar(self, current_stdout):
        """Generate a progress bar figure based on the current stdout."""
        matches = self._find_matches(self.pattern, current_stdout)
        percent = self._calculate_progress_percentage(matches)
        if percent > 0:
            return self._create_progress_bar_figure(percent)
        return self._create_default_figure()
