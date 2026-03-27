data:
  path: data/housing.csv
  target_column: price
  test_size: 0.2
  random_state: 42

model:
  name: linear

  params:
    linear: {}

    decision_tree:
      max_depth: 600
      min_samples_split: 400

    random_forest:
      n_estimators: 100
      max_depth: 40

artifacts:
  model_path: artifacts/house_price_linear.pkl

mlflow:
  experiment_name: house_price_linear_random_tree
  tracking_uri: sqlite:///mlflow.db