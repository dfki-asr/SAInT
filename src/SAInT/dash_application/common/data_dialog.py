import tkinter as tk
from tkinter import filedialog
from SAInT.common import makedirs

def _initialize_root() -> tk.Tk:
    """Initializes and returns a hidden Tkinter root window."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Ensure the window is on top
    root.update()  # Apply the topmost attribute
    return root

def _select_path(data_dir: str, title: str, mode: str) -> str:
    """
    Opens a Tkinter dialog to select a directory or file.

    :param data_dir: The initial directory to open in the dialog.
    :param title: The title of the dialog.
    :param mode: The mode of dialog ('directory' or 'file').

    :return: The selected path.
    """
    root = _initialize_root()
    if data_dir is not None:
        makedirs(data_dir)
    if mode == 'directory':
        path = filedialog.askdirectory(initialdir=data_dir, title=title)
    elif mode == 'file':
        path = filedialog.askopenfilename(initialdir=data_dir, title=title)
    else:
        raise ValueError("Invalid mode. Use 'directory' or 'file'.")
    root.destroy()
    return path

def ask_for_directory(data_dir: str, title: str) -> str:
    """
    Opens a Tkinter dialog to select a directory.

    :param data_dir: The initial directory to open in the dialog.
    :param title: The title of the dialog.

    :return: The selected directory path.
    """
    return _select_path(data_dir, title, 'directory')

def ask_for_file(data_dir: str, title: str) -> str:
    """
    Opens a Tkinter dialog to select a file.

    :param data_dir: The initial directory to open in the dialog.
    :param title: The title of the dialog.

    :return: The selected file path.
    """
    return _select_path(data_dir, title, 'file')
