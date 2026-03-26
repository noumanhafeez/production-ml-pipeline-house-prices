from dataclasses import dataclass
from typing import Dict, Any
import yaml


@dataclass
class DataConfig:
    path: str
    target_column: str
    test_size: float
    random_state: int


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Dict[str, Any]]


@dataclass
class ArtifactsConfig:
    model_path: str


@dataclass
class MLflowConfig:
    experiment_name: str
    tracking_uri: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    artifacts: ArtifactsConfig
    mlflow: MLflowConfig


def load_config(config_path: str = "config.yaml") -> Config:
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    return Config(
        data=DataConfig(**cfg["data"]),
        model=ModelConfig(**cfg["model"]),
        artifacts=ArtifactsConfig(**cfg["artifacts"]),
        mlflow=MLflowConfig(**cfg["mlflow"]),
    )