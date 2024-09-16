from SAInT.dash_application.dash_component import DashComponent, dcc

class DashInterval(DashComponent):
    def __init__(self, interval_in_ms, id):
        super().__init__(id=id)
        self.interval_in_ms = interval_in_ms
        self.n_intervals = 0

    def to_html(self, pixel_def):
        return dcc.Interval(
            id=self.id,
            interval=self.interval_in_ms,
            n_intervals=self.n_intervals
        )
