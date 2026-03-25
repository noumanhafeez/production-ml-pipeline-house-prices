# main.py
from models.train import train
from src.utils.logger import get_logger

# Create a logger for this main script
logger = get_logger("main", "logs/main.log")

if __name__ == "__main__":
    try:
        data_path = "data/housing.csv"  # replace with your dataset path
        target_column = "price"         # replace with your target column name

        logger.info(f"Starting main execution with dataset: {data_path} and target column: {target_column}")

        # Train the model
        model_pipeline = train(data_path, target_column)

        logger.info("Main execution completed successfully.")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise