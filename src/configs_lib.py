from dataclasses import dataclass, field

from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore
import hydra

from src.models.base_model import ModelConfig

from src.models.registry import MODEL_REGISTRY
from typing import List, Optional, Any, Dict

from omegaconf import DictConfig

@dataclass
class PipelineConfig:
    name: str
    dataset_id: str
    data_cleaning_strategy_steps: List[str]
    feature_engineering_strategy_steps: Dict[str, List[str]]
    feature_engineering_selected_features: Dict[str, List[str]]
    test_size: float = 0.2
    random_state: int = 1337

@dataclass
class BaseConfig:
    pipeline: PipelineConfig
    model_agg: Optional[Any] = None
    model_usr: Optional[Any] = None