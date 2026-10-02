import os
import sys
import logging

def _add_file_handler(logger: logging.Logger, log_file_path: str) -> None:
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)-20.20s | %(message)s"
    )
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler._paircheck_handler = True
    logger.addHandler(file_handler)

def _add_console_handler(logger: logging.Logger) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)-20.20s | %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler._paircheck_handler = True
    logger.addHandler(console_handler)

def get_logger(name: str = "PairCheck", log_filename: str = "PairCheck.log", console: bool = True, file: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers[:]:
        if getattr(handler, "_paircheck_handler", False):
            logger.removeHandler(handler)
            handler.close()

    if console:
        _add_console_handler(logger)
    if file:
        _add_file_handler(logger, f"logs/{log_filename}")

    return logger

