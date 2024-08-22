from utils.imports import *

def generate_path(path: str):
    """
    Generates the necessary directory structure for storing graphs, models, and results.

    This function checks if the specified base directory exists. If it does not exist,
    the function creates the base directory and subdirectories for storing graphs, models,
    and results.

    Args:
        path (str): The base directory path where the directories will be created.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + '/graphs')
        os.makedirs(path + '/models')
        os.makedirs(path + '/results')
