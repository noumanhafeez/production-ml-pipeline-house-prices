# src/report.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("report", "logs/report.log")

def plot_predictions(y_test, y_pred, save_path: Path = Path("artifacts/prediction_vs_actual.png")):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Predicted vs Actual Prices")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved Predicted vs Actual plot to {save_path}")

def save_metrics(score, save_path: Path = Path("artifacts/metrics.csv")):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    metrics = {"R2_score": score}
    pd.DataFrame([metrics]).to_csv(save_path, index=False)
    logger.info(f"Saved metrics to {save_path}")