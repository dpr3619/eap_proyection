"""This module provides standard funtionalities to create the loggers for the project"""
import logging
import os
import sys


def set_logger_handler_format(handler: logging.Handler):
    """Sets the required logging format.

    Args:
        handler (logging.Handler): Handler associated with logger.
    """
    format_logger = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s() - [%(levelname)s] - %(message)s"
    )
    handler.setFormatter(format_logger)


def set_logger(module_name: str = None) -> logging.Logger:
    """Creates and formats a Logger object.

    Args:
        module_name (str, optional): Name of the module, as specified by __name__. Defaults to None.

    Returns:
        logging.Logger: Logger object.
    """
    logger = logging.getLogger(module_name)
    logger.handlers = []  # <== Solution for AWS Lambda log duplication
    handler = logging.StreamHandler(sys.stdout)

    try:
        level = logging.getLevelName(os.getenv("LOGGING_LEVEL", "INFO"))
        logger.setLevel(level)
    except:
        logger.setLevel(logging.INFO)
        logger.warning("Could not find the LOGGIN_LEVEL in environmental variables")

    set_logger_handler_format(handler=handler)
    logger.addHandler(handler)
    logger.propagate = False

    return logger