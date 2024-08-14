import os
from pathlib import Path

def get_base_model_name(data_folder):
    return os.path.basename(data_folder)

def infer_folder_path_from_datasettings_path(data_settings_path):
    return os.path.abspath(os.path.dirname(data_settings_path))

def get_file_extension(filename: str):
    return os.path.splitext(filename)[1]

def infer_modelfolder_from_modelfilename(filename: str):
    return Path(os.path.abspath(os.path.dirname(filename))).parent
