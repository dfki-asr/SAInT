from collections import namedtuple
import json
from SAInT.common import rel_to_abs_path


DataSettings = namedtuple(
    'DataSettings', 'experiment_type, modi, data_folder, output_names, \
      include_input_features, exclude_input_features, mode, metric, normalization, batchsize, \
      output_folder, num_total, delimiter, augmented_features, verbose')


def handle_relative_paths(content, data_folder):
    for path_name in ["data_folder", "output_folder"]:
        if path_name == "output_folder":
            if content[path_name] == "":
                content[path_name] = data_folder.replace("/data/", "/outputs/")
        if content[path_name][0] == "/":
            # Linux abs path
            continue
        if content[path_name][1:3] == ":\\":
            # Windows abs path
            continue
        rel_path = "/".join([data_folder, content[path_name]])
        abs_path = rel_to_abs_path(rel_path)
        content[path_name] = abs_path + "/"
    return content


def json_to_data_settings(content, data_folder: str = None):
    if isinstance(content, str):
        content = json.loads(content)
    if data_folder is not None:
        content = handle_relative_paths(content, data_folder)
    data_settings = DataSettings(**content)
    return data_settings


def load_data_settings_file(data_settings_path, data_folder):
    data_settings = None
    with open(data_settings_path, "r") as file:
        content = json.load(file)
        data_settings = json_to_data_settings(content, data_folder)
        print(f"loaded data settings from file: {data_settings_path}")
    return data_settings


def save_data_settings_file(data_settings, data_settings_path):
    with open(data_settings_path, "w") as file:
        data_settings_dict = data_settings._asdict()
        json.dump(data_settings_dict, file, indent=4)
