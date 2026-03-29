from features.config import LOGGER_CONFIG
from features.pipeline import FeaturesPipeline
from utils.logger import configure_logger


def main():
    configure_logger(**LOGGER_CONFIG)
    FeaturesPipeline().run()


if __name__ == "__main__":
    main()

