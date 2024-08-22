from utils.imports import *
# used to track results
def initiate_log(PATH: str) -> logging.Logger:
    """
    Initializes the logging configuration and sets up both file and console logging.

    This function configures the logging system to write log messages to a file and
    to also output them to the console. The log messages will be formatted with timestamps
    and will include messages at the INFO level or higher.

    Args:
        PATH (str): The directory path where the log file will be saved. The log file
                    will be named "experiment.log".

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Configure logging to write to a file with specified formatting and level
    logging.basicConfig(
        filename=PATH + "experiment.log",
        format='%(asctime)s %(message)s',
        level=logging.INFO
    )

    # Set up logging to also output to the console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # Define a simpler format for console output
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)

    # Add the console handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get and return a logger instance with the module's name
    logger = logging.getLogger(__name__)
    return logger