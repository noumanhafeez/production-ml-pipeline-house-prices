import pandas as pd
import joblib
from pathlib import Path

from src.utils.logger import get_logger
from src.config import load_config

logger = get_logger("predict", "logs/predict.log")


def predict():
    """
    Load trained model and make predictions on new data.
    """
    try:
        logger.info("Starting prediction process...")

        # Load config
        config = load_config()

        # Load model
        model_path = Path(config.artifacts.model_path)
        logger.info(f"Loading model from: {model_path}")
        pipeline = joblib.load(model_path)

        # Load data
        data_path = Path(config.data.path)
        logger.info(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)

        logger.info(f"Data loaded. Shape: {data.shape}")

        # Predict
        predictions = pipeline.predict(data)
        logger.info("Predictions generated successfully.")

        # Save predictions
        output_path = Path("data/predictions.csv")
        data["Predicted_Price"] = predictions
        data.to_csv(output_path, index=False)

        logger.info(f"Predictions saved at: {output_path}")

        return predictions

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    predict()