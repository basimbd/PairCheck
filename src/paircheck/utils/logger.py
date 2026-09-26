import sys
import logging

def _add_file_handler(logger: logging.Logger, log_file_path: str) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(filename)-20.20s | %(message)s"
    )
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def _add_console_handler(logger: logging.Logger) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)-20.20s | %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def get_logger(name: str = "PairCheck", log_filename: str = "PairCheck.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    _add_console_handler(logger)
    _add_file_handler(logger, f"logs/{log_filename}")

    return logger

LOGGER = get_logger()
