from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

from src.config import load_config
from src.utils.logger import get_logger

app = Flask(__name__)
logger = get_logger("api", "logs/api.log")

# Load model once (VERY IMPORTANT for performance)
config = load_config()
model_path = Path(config.artifacts.model_path)

logger.info(f"Loading model from {model_path}")
model = joblib.load(model_path)


@app.route("/")
def home():
    return "ML Model API is running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        logger.info(f"Received input: {data}")

        # Convert JSON → DataFrame
        df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)