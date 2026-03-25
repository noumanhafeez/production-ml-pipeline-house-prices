# src/train.py
from pathlib import Path
import joblib
from src.data_loader import load_data
from src.data_splitter import split_data
from src.pipeline import create_pipeline
from src.utils.logger import get_logger
from src.report import plot_predictions, save_metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Create a logger for this module
logger = get_logger("train", "logs/train.log")


def train(path: str, target_col: str) :
    """
    Train the pipeline on the given dataset, save the trained model,
    generate visualization reports, and save metrics.
    Logs every major step.
    """
    try:
        logger.info(f"Starting training with dataset: {path} and target column: {target_col}")

        # Load data
        df = load_data(path)
        logger.info(f"Data loaded. Shape: {df.shape}")

        # Split data
        X_train, X_test, y_train, y_test = split_data(df, target_col)
        logger.info(f"Data split into train and test sets. "
                    f"X_train: {X_train.shape}, X_test: {X_test.shape}, "
                    f"y_train: {y_train.shape}, y_test: {y_test.shape}")

        # Create pipeline
        pipeline = create_pipeline()
        logger.info("Pipeline created successfully.")

        # Fit pipeline
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline training completed.")

        # Evaluate
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"R2 Score: {r2:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")

        # Generate visualization and metrics
        plot_predictions(y_test, y_pred)  # Save Predicted vs Actual plot
        save_metrics({
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        })

        # Save trained pipeline
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        model_path = artifacts_dir / "house_price_pipeline.pkl"
        joblib.dump(pipeline, model_path)
        logger.info(f"Trained model saved at: {model_path}")

        return pipeline

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise