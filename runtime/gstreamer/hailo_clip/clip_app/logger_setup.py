import logging

def setup_logger(name=__name__, level=logging.DEBUG):
    # Set up a logger with the specified name and level.
    # Args:
    #     name (str): The name of the logger. Defaults to the name of the current module.
    #     level (int): The initial logging level. Defaults to logging.DEBUG.
    # Returns:
    #     logger: The configured logger instance.

    # Get the logger with the specified name
    logger = logging.getLogger(name)
    # Set the logging level
    logger.setLevel(level)
    if not logger.handlers:
        # Create a console handler with the specified level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Add the formatter to the handler
        console_handler.setFormatter(formatter)
        # Add the handler to the logger
        logger.addHandler(console_handler)
    return logger

def set_log_level(logger, level):
    """
    Set the logging level for both the logger and its handlers.
    Args:
        logger: The logger instance whose level should be changed.
        level (int): The new logging level.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
