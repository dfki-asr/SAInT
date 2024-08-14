import re
import ast


def infer_data_type(value):
    if re.match(r'^-?\d+(\.|,)\d+((E|e)(-|\+)?\d+)?$', value):
        return "float"
    elif re.match(r'^-?\d+$', value):
        return "int"
    else:
        return "str"


def settings_value_str_to_dict(settings_value):
    # Define the pattern to match "feature=value"
    pattern = r'(\w+)\s*[:=]\s*((?:\[.*?\])|(?:\b\w+\b))(?=\s*(?:,|$))'
    # Use re.findall to find all matches of the pattern in the string
    matches = re.findall(pattern, settings_value)
    # Create a dictionary to store the feature-value pairs
    result = {}
    for match in matches:
        key, value = match
        try:
            value = ast.literal_eval(value.strip())
        except ValueError:
            pass
        result[key] = value
    return result
