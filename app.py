from flask import Flask, request, render_template
import joblib
import pandas as pd
from pathlib import Path

from src.config import load_config
from src.utils.logger import get_logger

app = Flask(__name__)
logger = get_logger("api", "logs/api.log")

# Load model
config = load_config()
model_path = Path(config.artifacts.model_path)
logger.info(f"Loading model from {model_path}")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        input_data = {
            "area": int(data["area"]),
            "bedrooms": int(data["bedrooms"]),
            "bathrooms": int(data["bathrooms"]),
            "stories": int(data["stories"]),
            "mainroad": data["mainroad"],
            "guestroom": data["guestroom"],
            "basement": data["basement"],
            "hotwaterheating": data["hotwaterheating"],
            "airconditioning": data["airconditioning"],
            "parking": int(data["parking"]),
            "prefarea": data["prefarea"],
            "furnishingstatus": data["furnishingstatus"],
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        # 👉 Send result to new page
        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return render_template("result.html", prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)