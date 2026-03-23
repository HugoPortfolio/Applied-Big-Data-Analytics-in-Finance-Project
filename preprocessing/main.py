from config import PREPROCESSING_LOG_PATH
from logger import configure_logger
from pipeline import PreprocessingPipeline


def main():
    configure_logger(PREPROCESSING_LOG_PATH)
    pipeline = PreprocessingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()