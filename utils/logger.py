import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ConsoleOnlyImportantFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return (
            "run_config" in msg
            or "period_start" in msg
            or "batch_flush" in msg
            or "shard_completed" in msg
            or "scraping_completed" in msg
            or record.levelno >= logging.ERROR
        )


def configure_logger(
    log_path: Path,
    *,
    logger_name: str | None = None,
    level: int = logging.INFO,
    use_root: bool = True,
    propagate: bool = False,
    console_filter: logging.Filter | None = None,
    rotating_file: bool = True,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    console_with_filename_lineno: bool = False,
):
    logger = logging.getLogger() if use_root else logging.getLogger(logger_name)

    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(level)
    logger.propagate = propagate

    log_path.parent.mkdir(parents=True, exist_ok=True)

    console_fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
        if console_with_filename_lineno
        else "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_fmt = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

    console_formatter = logging.Formatter(console_fmt)
    file_formatter = logging.Formatter(file_fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    if console_filter is not None:
        console_handler.addFilter(console_filter)

    if rotating_file:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")

    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)