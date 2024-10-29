from dataclasses import dataclass,field
from typing import List, Tuple

@dataclass
class DataIngestorConfig:
    ARTIFACT_DIR: str
    DATASET_NAME: str
    BUCKET_NAME: str
    CLOUD_DIR: str

@dataclass
class PreProcessorConfig:
    ARTIFACT_DIR: str
    IMAGE_SIZE: int
    MEAN: List[float]
    STD: List[float]
    BRIGHTNESS: float
    CONTRAST: float
    SATURATION: float
    HUE: float
    PROB: float

@dataclass
class DataModuleConfig:
    DATA_DIR: str
    PREPROCESSOR_DIR: str
    TRAIN_SIZE: float
    IMAGE_SIZE: 224
    ANCHORS: List[List[Tuple[float, float]]]

@dataclass
class LossConfig:
    ANCHORS: List[List[Tuple[float, float]]]
    LAMBDA_OBJ: float
    LAMBDA_NOOBJ: float
    LAMBDA_COORD: float
    LAMBDA_CLASS: float

@dataclass
class ModelConfig:
    IN_CHANNELS: int
    NUM_CLASSES: int

@dataclass
class TrainingConfig:
    NUM_EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float

@dataclass
class TrainerConfig:
    ARTIFACT_DIR: str
    DATA_MODULE_CONFIG: DataModuleConfig
    LOSS_CONFIG: LossConfig
    MODEL_CONFIG: ModelConfig
    TRAINING_CONFIG: TrainingConfig

@dataclass
class EvaluatorConfig:
    DATA_MODULE_CONFIG: DataModuleConfig
    BATCH_SIZE: int
    LOSS_CONFIG: LossConfig
    PREPROCESSOR_PATH: str
    KERAS_MODEL_PATH: str
    SAVED_MODEL_PATH: str