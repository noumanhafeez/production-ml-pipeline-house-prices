from models.train import train
from src.config import load_config
from src.utils.logger import get_logger

logger = get_logger("main", "logs/main.log")

if __name__ == "__main__":
    try:
        config = load_config()

        logger.info("Config loaded successfully")

        train(config)

        logger.info("Execution completed successfully")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise