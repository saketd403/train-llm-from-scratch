
import logging

def setup_logger(name: str):

    # Create logger
    logger = logging.getLogger(name)

    # Set the logging level (Optional)
    logger.setLevel(logging.DEBUG)  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()

    # Define the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    return logger
