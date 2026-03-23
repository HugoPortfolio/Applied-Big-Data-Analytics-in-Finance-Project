from scraping.config import LOGGER_CONFIG
from scraping.scraper import KoyfinScraper
from utils.logger import configure_logger, ConsoleOnlyImportantFilter, get_logger


def main():
    configure_logger(
        **LOGGER_CONFIG,
        console_filter=ConsoleOnlyImportantFilter(),
    )
    logger = get_logger("koyfin_scraper")
    scraper = KoyfinScraper(logger=logger)
    scraper.run()


if __name__ == "__main__":
    main()