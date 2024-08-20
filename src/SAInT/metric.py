from collections import OrderedDict
from SAInT.metrics import regression
from SAInT.metrics import classification

def get_functions_dict(module):
    # Use dir() to get a list of attributes from the module
    attributes = dir(module)
    # Filter out attributes that are callable (functions)
    functions = {name: getattr(module, name) for name in attributes if callable(getattr(module, name))}
    return functions

def get_ordered_functions_dict(module):
    # Retrieve the __all__ list if it exists
    all_names = getattr(module, '__all__', None)
    # Use an ordered dictionary to preserve the order
    functions = OrderedDict()
    if all_names:
        for name in all_names:
            if callable(getattr(module, name, None)):
                functions[name] = getattr(module, name)
    return functions

# Create dictionaries for functions in the regression and classification modules
regression_functions_dict = get_ordered_functions_dict(regression)
classification_functions_dict = get_ordered_functions_dict(classification)
metric_functions_dict = regression_functions_dict | classification_functions_dict

def get_metric(metric_name="mae", args: dict = None):
    if metric_name not in metric_functions_dict.keys():
        raise RuntimeError(f"Unknown metric {metric_name}!")
    return metric_functions_dict[metric_name]

def get_list_of_supported_standard_metrics():
    return list(metric_functions_dict.keys())

def get_list_of_standard_metric_functions():
    return list(metric_functions_dict.values())
