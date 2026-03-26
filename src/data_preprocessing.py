import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from .utils.logger import get_logger

logger = get_logger("preprocess", "logs/preprocess.log")

def preprocess() -> ColumnTransformer:
    """
    Preprocess the dataset using ColumnTransformer.

    - Binary features (yes/no) -> 0/1
    - Multi-category features -> one-hot encoding
    - Numeric features -> passed through
    Returns a ColumnTransformer and ready to fit or transform.

    """
    try:
        # Define columns
        binary_cols = ['mainroad', 'guestroom', 'basement',
                       'hotwaterheating', 'airconditioning', 'prefarea']
        multi_cat_cols = ['furnishingstatus']

        logger.info(f"Starting preprocessing. Binary columns: {binary_cols}, Multi-category columns: {multi_cat_cols}")

        # Binary encoder (yes->1, no->0)
        binary_encoder = OrdinalEncoder(categories=[['no', 'yes']] * len(binary_cols))

        # One-hot encoder
        onehot_encoder = OneHotEncoder(drop='first', dtype=int)

        # ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('binary', binary_encoder, binary_cols),
                ('onehot', onehot_encoder, multi_cat_cols)
            ],
            remainder='passthrough'  # keep numeric columns unchanged
        )

        logger.info("Preprocessor created successfully.")

        return preprocessor

    except KeyError as e:
        logger.error(f"Column not found in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise