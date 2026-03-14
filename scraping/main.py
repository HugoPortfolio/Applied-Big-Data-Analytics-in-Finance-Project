from scraping.logger_setup import build_logger
from scraping.scraper import KoyfinScraper


def main():
    logger = build_logger()
    scraper = KoyfinScraper(logger=logger)
    scraper.run()


if __name__ == "__main__":
    main()