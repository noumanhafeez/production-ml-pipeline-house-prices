# src/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from .data_preprocessing import preprocess
from .utils.logger import get_logger

# Create a logger for this module
logger = get_logger("pipeline", "logs/pipeline.log")


def create_pipeline() -> Pipeline:
    """
    Create and return a sklearn pipeline with preprocessing and Linear Regression.
    Logs the pipeline creation steps.
    Returns:
        sklearn.pipeline.Pipeline
    """
    try:
        logger.info("Starting pipeline creation.")

        # Create preprocessor
        preprocessor = preprocess()
        logger.info("Preprocessor created successfully.")

        # Create the full pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])

        logger.info("Pipeline created successfully with LinearRegression as regressor.")

        return pipeline

    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise