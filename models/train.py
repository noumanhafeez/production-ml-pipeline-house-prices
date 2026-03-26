from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from src.data_loader import load_data
from src.data_splitter import split_data
from src.pipeline import create_pipeline
from src.report import plot_predictions, save_metrics
from src.utils.logger import get_logger

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

logger = get_logger("train", "logs/train.log")


def train(config):
    try:
        logger.info("Starting training...")

        # MLflow setup
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

        with mlflow.start_run():

            # Load data
            df = load_data(config.data.path)
            logger.info(f"Dataset shape: {df.shape}")

            # Split
            X_train, X_test, y_train, y_test = split_data(
                df,
                config.data.target_column,
                test_size=config.data.test_size,
                random_state=config.data.random_state
            )
            logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

            # Create pipeline
            logger.info("Creating pipeline...")
            pipeline = create_pipeline(
                config.model.name,
                config.model.params.get(config.model.name, {})
            )
            logger.info(f"Pipeline created successfully")

            # Train
            logger.info(f"Training model...")
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred = pipeline.predict(X_test)


            # Metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Generate visualization and metrics
            logger.info("Generating evaluation artifacts...")
            plot_predictions(y_test, y_pred)  # Save Predicted vs Actual plot
            save_metrics({
                "r2": r2,
                "mae": mae,
                "rmse": rmse
            })

            # Log params
            mlflow.log_params(config.model.params)
            mlflow.log_param("model_name", config.model.name)
            mlflow.log_param("target_column", config.data.target_column)
            mlflow.log_param("test_size", config.data.test_size)
            mlflow.log_param("random_state", config.data.random_state)


            # Log metrics
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)


            # Log model
            logger.info("Logging model to MLflow...")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=config.model.name
            )

            # Save model locally
            model_path = Path(config.artifacts.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)

            logger.info(f"Model saved at {model_path}")
            logger.info(f"R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            return pipeline

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise