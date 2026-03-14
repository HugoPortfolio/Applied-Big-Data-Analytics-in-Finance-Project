import logging

from scraping.config import LOG_DIR


LOG_PATH = LOG_DIR / "koyfin_scraping.log"


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


def build_logger(name="koyfin_scraper"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(ConsoleOnlyImportantFilter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger