from preprocessing.config import LOGGER_CONFIG
from preprocessing.pipeline import PreprocessingPipeline
from utils.logger import configure_logger


def main():
    configure_logger(**LOGGER_CONFIG)
    pipeline = PreprocessingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()