from scoring.config import LOGGER_CONFIG
from scoring.pipeline import ScoringPipeline
from utils.logger import configure_logger


def main():
    configure_logger(**LOGGER_CONFIG)
    pipeline = ScoringPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
