import re

def scale_html(html_content, scale=1.0, max_width=None):
    """
    Scale the HTML content for better display.

    :param html_content: The HTML content to be scaled.
    :param scale: The scaling factor.
    :param max_width: The maximum width for the scaled content.

    :return: The scaled HTML content.
    """
    style = f"transform: scale({scale}); transform-origin: top left;"
    if max_width:
        style += f" max-width: {max_width}px; overflow-x: hidden;"
    return html_content.replace("<body>", f"<body style=\"{style}\">")

def check_for_sensitive_features(top_features):
    sensitive_features_regex = re.compile(
        r'([Aa]ge|[Ss]ex(?:_[a-zA-Z]+)?|[Rr]ace(?:[Dd]esc(?:_[A-Za-z\s]+)?)?|[Gg]ender(?:ID)?)'
    )
    # Function to check if a feature name matches the sensitive pattern
    def is_feature_name_match(feature_name):
        return bool(sensitive_features_regex.match(feature_name))
    # List to store detected sensitive features
    sensitive_features = [f for f in top_features if is_feature_name_match(f)]
    # Log or handle sensitive features if found
    if sensitive_features:
        print(f"Found sensitive features: {sensitive_features}")
    return sensitive_features
