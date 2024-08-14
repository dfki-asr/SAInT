from fastai.callback.core import Callback, CancelFitException
import os

class StopTrainingCallback(Callback):
    def __init__(self, stop_signal_file="stop_signal.txt"):
        self.stop_signal_file = stop_signal_file

    def before_batch(self):
        if os.path.exists(self.stop_signal_file):
            print("Training stopped by user")
            #self.run = False
            raise CancelFitException()

    def after_fit(self):
        # Remove the stop signal file after training
        if os.path.exists(self.stop_signal_file):
            os.remove(self.stop_signal_file)
