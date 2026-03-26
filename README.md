# 🏠 House Price Prediction

A production-style machine learning pipeline for predicting house prices using Linear Regression. Built with scikit-learn, this project follows clean architecture principles with modular components, structured logging, and artifact management.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   └── housing.csv                  # Input dataset
│
├── src/
│   ├── data_loader.py               # CSV ingestion
│   ├── data_splitter.py             # Train/test split
│   ├── data_preprocessing.py        # Feature encoding & transformation
│   ├── pipeline.py                  # sklearn Pipeline builder
│   ├── report.py                    # Metrics & plot generation
│   └── utils/
│       └── logger.py                # Centralized logging utility
│
├── models/
│   └── train.py                     # Training orchestration
│
├── artifacts/
│   ├── house_price_pipeline.pkl     # Saved trained model
│   ├── metrics.csv                  # Evaluation metrics
│   └── prediction_vs_actual.png     # Visualization plot
│
├── logs/                            # Per-module log files
├── main.py                          # Entry point
└── README.md
```

---

## ⚙️ How It Works

The pipeline follows a clean, sequential flow:

```
CSV Data → Load → Split → Preprocess → Train → Evaluate → Save Artifacts
```

### Feature Engineering

| Feature Type | Columns | Encoding |
|---|---|---|
| Binary (yes/no) | `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea` | Ordinal (0/1) |
| Multi-category | `furnishingstatus` | One-Hot (drop first) |
| Numeric | All remaining | Passthrough |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

### Run Training

```bash
python main.py
```

Configure the dataset path and target column directly in `main.py`:

```python
data_path = "data/housing.csv"
target_column = "price"
```

---

## 📊 Outputs

After training, the following artifacts are saved to the `artifacts/` directory:

| Artifact | Description |
|---|---|
| `house_price_pipeline.pkl` | Serialized trained sklearn pipeline |
| `metrics.csv` | R², MAE, and RMSE scores |
| `prediction_vs_actual.png` | Scatter plot of predicted vs actual prices |

---

## 📈 Evaluation Metrics

The model is evaluated on a held-out test set (20% of data) using:

- **R² Score** — Proportion of variance explained
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error

---

## 🧩 Module Overview

| Module | Responsibility |
|---|---|
| `data_loader.py` | Reads CSV with error handling and logging |
| `data_splitter.py` | Stratified 80/20 train-test split |
| `data_preprocessing.py` | Builds a `ColumnTransformer` for encoding |
| `pipeline.py` | Composes preprocessor + `LinearRegression` into a `Pipeline` |
| `train.py` | Orchestrates end-to-end training, evaluation, and artifact saving |
| `report.py` | Saves metrics to CSV and generates prediction plots |
| `logger.py` | Configures per-module file loggers |

---

## 📋 Requirements

```
pandas
scikit-learn
matplotlib
joblib
numpy
```

> Add these to a `requirements.txt` file at the project root.

---

## 🪵 Logging

Each module writes to its own log file under `logs/`:

```
logs/
├── main.log
├── train.log
├── data_loader.log
├── data_splitter.log
├── preprocess.log
├── pipeline.log
└── report.log
```

---

## 🔭 Future Improvements

- [ ] Add hyperparameter tuning with `GridSearchCV`
- [ ] Experiment with ensemble models (Random Forest, XGBoost)
- [ ] Add cross-validation support
- [ ] Expose training via CLI with `argparse`
- [ ] Add unit tests with `pytest`
- [ ] Containerize with Docker

---

## 📄 License

This project is licensed under the MIT License.