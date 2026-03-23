import logging
from logging.handlers import RotatingFileHandler


def configure_logger(log_path, level=logging.INFO):
    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.setLevel(level)

    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name):
    return logging.getLogger(name)