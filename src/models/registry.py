from src.models.aggregated.kNN.model import kNNModelConfig, kNNModelBuilding
from src.models.aggregated.XGBRegressor.model import XGBRegressorModelConfig, XGBRegressorModelBuilding

from src.models.user.bayesian.model import BayesianModelConfig, BayesianModelBuilding



MODEL_REGISTRY = {
    "agg": {
        "XGBRegressor": {
        "config": XGBRegressorModelConfig,
        "model_class": XGBRegressorModelBuilding
        },
        "kNN": {
            "config": kNNModelConfig,
            "model_class": kNNModelBuilding
        }    
    },
    "usr": {
        "bayesian": {
        "config": BayesianModelConfig,
        "model_class": BayesianModelBuilding
        },

    }    
}