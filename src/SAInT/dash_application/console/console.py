import io
import sys
import logging
from SAInT.dash_application.common.dash_functions import get_pressed_buttons

# Disable Flask's log messages
logging.getLogger('werkzeug').setLevel(logging.ERROR)

class Console:
    def __init__(self):
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer

    def clear(self):
        self.stdout_buffer.seek(0)
        self.stdout_buffer.truncate(0)
        self.stderr_buffer.seek(0)
        self.stderr_buffer.truncate(0)

    def update(self):
        changed_id = get_pressed_buttons()
        current_std_output = self.stdout_buffer.getvalue()
        current_error_output = self.stderr_buffer.getvalue()
        current_output = current_std_output + current_error_output

        if "clear_console_button.n_clicks" in changed_id:
            self.clear()
        return current_output
