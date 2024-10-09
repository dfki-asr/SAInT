import os
import json
import csv
import shutil
from timeit import default_timer as timer
from pathlib import Path
from SAInT.common import exists
from SAInT.trainer import Trainer
from SAInT.dash_application.common.data_dialog import ask_for_directory, ask_for_file
from SAInT.dash_application.settings.app_settings import load_app_settings_file, save_app_settings_file
from SAInT.data_settings import load_data_settings_file, save_data_settings_file
from SAInT.dash_application.common.config_conversion import infer_data_type
from SAInT.dash_application.components import DashRadioButton, DashChecklist, DashDiv, DashNewline, DashGrid
import importlib.resources as pkg_resources
from SAInT.templates import __name__ as template_module

def get_values_by_id(elements, target_id, target_type):
    # Check if the element is a list
    if isinstance(elements, list):
        # Recursively check each item in the list
        for element in elements:
            result = get_values_by_id(element, target_id, target_type)
            if result is not None:
                return result
    # Check if the current element is a dictionary with 'props'
    elif isinstance(elements, dict) and 'props' in elements:
        # Check if 'id' in props matches the target_id and type is target_type
        if elements['props'].get('id') == target_id and elements.get('type') == target_type:
            # Return the 'value' in props if it exists
            return elements['props'].get('value')
        # Recursively check children if they exist
        if 'children' in elements['props']:
            children = elements['props']['children']
            return get_values_by_id(children, target_id, target_type)
    return None

class DataHandler:
    def __init__(self, application):
        self.application = application
        self.all_feature_info = {}

    ########################
    # Data Loading
    ########################

    @staticmethod
    def _get_default_dir(subfolder):
        """Get the default directory for the specified subfolder."""
        return str(Path(os.path.abspath(os.path.curdir)).parent / subfolder)

    @staticmethod
    def _get_default_data_dir():
        """Get the default data directory."""
        data_dir = DataHandler._get_default_dir("data")
        if exists(data_dir):
            return data_dir
        return None

    @staticmethod
    def _get_default_output_dir():
        """Get the default output directory."""
        return DataHandler._get_default_dir("outputs")

    def _copy_settings_template_file(self, settings_file: str, file_name: str):
        """Copy the settings template file to the specified location."""
        with pkg_resources.path(template_module, f"{file_name}.json") as settings_filepath:
            shutil.copy(str(settings_filepath), settings_file)
            print(f"Copied template file {settings_filepath} to {settings_file}.")

    def _load_settings(self, settings_file: str, file_name: str):
        """Load settings from a specified file."""
        if file_name == "app_settings":
            settings = load_app_settings_file(settings_file)
            if settings is None:
                self._copy_settings_template_file(settings_file, file_name)
                settings = load_app_settings_file(settings_file)
        elif file_name == "data_settings":
            folder_name = os.path.abspath(os.path.join(settings_file, os.pardir))
            settings = load_data_settings_file(settings_file, folder_name)
            if settings is None:
                self._copy_settings_template_file(settings_file, file_name)
                settings = load_data_settings_file(settings_file, folder_name)
        else:
            raise RuntimeError(f"Unknown file {file_name}")
        return settings

    def _read_row_of_csv(self, csv_path, row_number):
        """Read a specific row from a CSV file."""
        def read_line(csv_path, encoding):
            with open(csv_path, 'r', encoding=encoding) as file:
                for _ in range(row_number):
                    row_data = file.readline()
            return row_data

        try:
            return read_line(csv_path, 'utf-8')
        except UnicodeDecodeError:
            return read_line(csv_path, 'ISO-8859-1')

    def _auto_infer_delimiter_from_csv_file(self, csv_first_row):
        """Automatically infer the delimiter from the first row of a CSV file."""
        if csv_first_row:
            if ';' in csv_first_row:
                return ";"
            if ',' in csv_first_row:
                return ","
            print('Could not infer delimiter.')
        return None

    def _separate_csv_values(self, row_str, delimiter):
        """Separate CSV values using the specified delimiter."""
        return next(csv.reader([row_str], delimiter=delimiter, quotechar='"'))

    def _auto_infer_data_types_from_csv_file(self, csv_second_row, delimiter):
        """Automatically infer data types from the second row of a CSV file."""
        if csv_second_row:
            row = self._separate_csv_values(csv_second_row.replace("\n", ""), delimiter)
            return [infer_data_type(col) for col in row]
        return None

    def _auto_infer_feature_names_from_csv_file(self, csv_first_row, delimiter):
        """Automatically infer feature names from the first row of a CSV file."""
        if csv_first_row:
            return self._separate_csv_values(csv_first_row.replace("\n", ""), delimiter)
        return None

    def _auto_infer_modi_and_rename_files(self, folder_path, csv_files):
        """Automatically infer modi and rename files in the folder."""
        if not csv_files:
            raise RuntimeError("No CSV files found -> can't infer modi!")

        if len(csv_files) == 1:
            modi = ["total"]
            src = os.path.join(folder_path, csv_files[0])
            dst = os.path.join(folder_path, "total_data.csv")
            os.rename(src, dst)
        else:
            modi = []
            for file in csv_files:
                for mode in ["train", "valid", "test"]:
                    if mode in file:
                        modi.append(mode)
                        src = os.path.join(folder_path, file)
                        dst = os.path.join(folder_path, f"{mode}_data.csv")
                        os.rename(src, dst)
        return modi

    def _list_csv_files_in_folder(self, folder_path):
        """List all CSV files in the specified folder."""
        csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        if not csv_files:
            raise RuntimeError(f"No CSV files found in directory: {folder_path}.")
        return csv_files

    def _infer_information_from_csv_data(self, folder_path):
        """Infer information from CSV data in the specified folder."""
        csv_files = self._list_csv_files_in_folder(folder_path)
        if not csv_files:
            return None

        first_csv_path = os.path.join(folder_path, csv_files[0])
        csv_first_row = self._read_row_of_csv(first_csv_path, 1)
        csv_second_row = self._read_row_of_csv(first_csv_path, 2)

        delimiter = self._auto_infer_delimiter_from_csv_file(csv_first_row)
        feature_names = self._auto_infer_feature_names_from_csv_file(csv_first_row, delimiter)
        data_types = self._auto_infer_data_types_from_csv_file(csv_second_row, delimiter)
        modi = self._auto_infer_modi_and_rename_files(folder_path, csv_files)
        output_folder = folder_path.replace("/data/", "/outputs/")

        print(f"modi: {modi}, delimiter: '{delimiter}', feature_names: {feature_names}, output_folder: {output_folder}")

        return {
            "modi": modi,
            "feature_names": feature_names,
            "data_types": data_types,
            "output_folder": output_folder,
            "delimiter": delimiter
        }

    def load_app_settings(self, settings_file: str):
        """Load application settings from the specified file."""
        return self._load_settings(settings_file, "app_settings")

    def load_data_settings(self, settings_file: str):
        """Load data settings from the specified file."""
        return self._load_settings(settings_file, "data_settings")

    def data_settings_to_json(self, settings):
        """Convert data settings to a JSON string."""
        return json.dumps(settings._asdict(), indent=4)

    def save_app_settings(self, settings, data_settings_path: str = None):
        """Save application settings to the specified file."""
        save_app_settings_file(settings=settings, settings_file=data_settings_path)

    def load_data_settings_and_update_application(self, folder_path: str, data_settings_path: str = None):
        """Load data settings and update the application accordingly."""
        start = timer()
        self.application.trainer = None
        self.application.reset()

        settings_file = os.path.join(folder_path, "app_settings.json")
        self.application.settings = self.load_app_settings(settings_file)

        copy_template = False
        if data_settings_path is None:
            data_settings_path = os.path.join(folder_path, "data_settings.json")
            if not os.path.exists(data_settings_path):
                copy_template = True

        info = self._infer_information_from_csv_data(folder_path)
        self.all_feature_info = {
            "feature_names": info["feature_names"],
            "data_types": info["data_types"]
        }

        if copy_template:
            self._copy_settings_template_file(settings_file=data_settings_path, file_name="data_settings")
            data_settings = self.load_data_settings(data_settings_path)
            data_settings = data_settings._replace(
                modi=info["modi"],
                delimiter=info["delimiter"],
                output_folder=info["output_folder"]
            )
            save_data_settings_file(data_settings=data_settings, data_settings_path=data_settings_path)

        self.application.interactive_plot.dataset_selection = self.application.settings.ds_for_model_selection
        self.application.trainer = Trainer(data_folder=folder_path, data_settings_path=data_settings_path)
        print(f"load data settings took {(timer() - start):.2f} s.")
        return self.application

    def load_data(self):
        """Load data using the settings specified in the application."""
        start = timer()
        if self.application.trainer and self.application.settings:
            self.application.trainer.load_data(
                procs=self.application.settings.procs,
                do_one_hot_encoding=self.application.settings.do_one_hot_encoding,
                dtype=self.application.settings.dtype,
                valid_frac=float(self.application.settings.valid_frac),
                test_frac=float(self.application.settings.test_frac)
            )
            if self.application.trainer.data_settings:
                if self.application.trainer.data_settings.verbose:
                    self.create_histograms(do_save=True, do_show=False)
            print(f"load data took {(timer() - start):.2f} s.")

    def create_histograms(self, do_save: bool = True, do_show: bool = True):
        """Create histograms of the data."""
        self.application.trainer.create_histograms(do_save=do_save, do_show=do_show)

    def select_folder(self):
        """Prompt the user to select a data folder."""
        data_dir = self._get_default_data_dir()
        title = "Select Data Folder"
        return ask_for_directory(data_dir=data_dir, title=title)

    def select_file(self):
        """Prompt the user to select a data_settings.json file."""
        data_dir = self._get_default_data_dir()
        title = "Select data_settings.json File"
        return ask_for_file(data_dir=data_dir, title=title)

    ########################
    # Feature Selection
    ########################

    def _get_input_features_from_popup_body(self, popup_body):
        """Get input features from the popup body."""
        return get_values_by_id(popup_body, "inputnames_checklist", "Checklist")

    def _get_output_features_from_popup_body(self, popup_body):
        """Get output features from the popup body."""
        return get_values_by_id(popup_body, "outputnames_checklist", "Checklist")

    @staticmethod
    def _remove_string_quotes(features):
        """Remove quotes from the feature names."""
        return [feature.replace("\"", "") for feature in features]

    def _from_data_settings(self, features):
        """Process features from data settings."""
        return self._remove_string_quotes(features) if features else []

    def open_feature_selection_popup(self):
        """Open the feature selection popup."""
        feature_names = self.all_feature_info["feature_names"]
        data_types = self.all_feature_info["data_types"]
        options = self._remove_string_quotes(feature_names)

        data_settings = self.load_data_settings(self.application.trainer.data_settings_path)

        selected_output = self._from_data_settings(data_settings.output_names)
        included_input = self._from_data_settings(data_settings.include_input_features)
        excluded_input = self._from_data_settings(data_settings.exclude_input_features)
        if not included_input:
            included_input = options
        selected_input = [feature for feature in included_input if feature not in excluded_input and feature not in selected_output]

        options_with_dtype = [{'label': f"{option} ({dtype})", 'value': option} for option, dtype in zip(options, data_types)]

        mode_radio_button = DashRadioButton(name="Mode",
                                            options=["regression", "classification"],
                                            default_value=data_settings.mode,
                                            id="mode_radiobutton")
        normalization_radio_button = DashRadioButton(name="Normalization",
                                            options=["none", "mean_std", "min_max"],
                                            default_value=data_settings.normalization,
                                            id="normalization_radiobutton")
        input_checklist = DashChecklist(name="Input Features",
                                        options=options_with_dtype,
                                        default_value=selected_input,
                                        id="inputnames_checklist",
                                        inline=False)
        output_checklist = DashChecklist(name="Output Features",
                                         options=options_with_dtype,
                                         default_value=selected_output,
                                         id="outputnames_checklist",
                                         inline=False)
        pixel_def = self.application.pixel_definitions
        if pixel_def is None:
            raise RuntimeError("Pixel Definition error!")
        body = [
            DashGrid(id="feature_in_out_checkbox_div", item_values=[
                DashDiv(id="feature_popup_mode_div", content=[mode_radio_button,
                                                              DashNewline()]),
                DashDiv(id="feature_popup_norm_div", content=[normalization_radio_button,
                                                              DashNewline()]),
                DashDiv(id="feature_in_checkbox_div", content=[input_checklist,
                                                               DashNewline()]),
                DashDiv(id="feature_out_checkbox_div", content=[output_checklist,
                                                                DashNewline()])
            ], item_widths=[6, 6, 6, 6]).to_html(pixel_def)
        ]
        self.application.feature_selection_popup.set_content(body)
        self.application.feature_selection_popup.open()

    def read_update_and_save_data_settings_file(self, input_names, output_names, mode, normalization):
        """Read, update, and save data settings file."""
        data_settings_path = self.application.trainer.data_settings_path
        data_settings = self.load_data_settings(data_settings_path)
        data_settings = data_settings._replace(
            include_input_features=input_names,
            output_names=output_names,
            mode=mode,
            normalization=normalization
        )
        save_data_settings_file(data_settings=data_settings, data_settings_path=data_settings_path)

    def extract_features_from_popup_body(self, popup_body):
        """Extract features from the popup body."""
        inputnames_checklist_values = self._get_input_features_from_popup_body(popup_body)
        outputnames_checklist_values = self._get_output_features_from_popup_body(popup_body)

        output_names = [str(value) for value in outputnames_checklist_values]
        if len(output_names) < 1:
            raise RuntimeError("No output features defined!")

        input_names = [str(value) for value in inputnames_checklist_values]
        if len(input_names) < 1:
            raise RuntimeError("No input features defined!")
        if any(name in input_names for name in output_names):
            raise RuntimeError("An output is also part of the input features!")

        data_types = self.all_feature_info["data_types"]
        feature_names = self.all_feature_info["feature_names"]

        output_indices = [feature_names.index(output_name) for output_name in output_names]
        output_data_types = [data_types[index] for index in output_indices]
        print("outputs: ", [f"{name} ({dtype})" for name, dtype in zip(output_names, output_data_types)])
        return input_names, output_names

    def get_data_info(self):
        """Get data information."""
        if not self.application.trainer:
            return ""

        trainer = self.application.trainer
        dataset_dict = trainer.dataloader.datasets
        dataset_names = [mode for mode, dataset in dataset_dict.items() if dataset]
        datasets = [mode for mode in dataset_names if dataset_dict[mode].num_samples > 0]
        num_datasets = len(datasets)
        num_inputs = len(trainer.input_names)
        if num_inputs < 1:
            raise RuntimeError("No input features defined!")
        inputs_str = ", ".join(trainer.input_names)
        num_outputs = len(trainer.target_names)
        if num_outputs < 1:
            raise RuntimeError("No output features defined!")
        outputs_str = ", ".join(trainer.target_names)
        modi_str = ", ".join(datasets)
        data_folder = self.application.data_folder
        return f"{num_datasets} datasets from {data_folder}: {modi_str},\n{num_inputs} inputs: {inputs_str},\n{num_outputs} outputs: {outputs_str}"

    def save_to_data_settings_file(self, popup_body):
        """Save features and mode to the data settings file."""
        input_names, output_names = self.extract_features_from_popup_body(popup_body)
        mode = get_values_by_id(popup_body, "mode_radiobutton", "RadioItems")
        normalization = get_values_by_id(popup_body, "normalization_radiobutton", "RadioItems")
        self.read_update_and_save_data_settings_file(input_names, output_names, mode, normalization)
