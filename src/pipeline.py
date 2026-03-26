# src/pipeline.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from .data_preprocessing import preprocess
from .utils.logger import get_logger

# Create a logger for this module
logger = get_logger("pipeline", "logs/pipeline.log")

MODEL_MAPPING = {
    "linear": LinearRegression,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor
}


def create_pipeline(model_name: str, model_params: dict) -> Pipeline:
    """
    Create and return a sklearn pipeline with preprocessing and selected models.
    Logs the pipeline creation steps.
    Returns:
        sklearn.pipeline.Pipeline
    """
    try:
        logger.info(f"Creating pipeline with model: {model_name}")

        if model_name not in MODEL_MAPPING:
            raise ValueError(f"Invalid model: {model_name}")

        # Create preprocessor
        preprocessor = preprocess()
        logger.info("Preprocessor created successfully.")

        params = model_params.get(model_name, {})
        model = MODEL_MAPPING[model_name](**params)

        # Create the full pipeline
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        logger.info(f"Pipeline created with params: {params}")

        return pipeline

    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise