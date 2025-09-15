import os
import shutil

from omegaconf import MISSING, OmegaConf
from hydra.utils import instantiate
import hydra

from src.configs_lib import BaseConfig

from pipelines.model_evaluation import ModelEvaluationPipeline


import sys
import logging

from src.common.utils import setup_logger

from metaflow import Runner
import json


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(cfg: BaseConfig) -> None:
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    main_log_file = os.path.join(output_dir, "main.log")
    flow_log_file = os.path.join(output_dir, "flow.log")

    
    setup_logger(log_file=main_log_file) 
    logger = logging.getLogger(__name__)
    logger.info("Logging is working.")

    if cfg.pipeline.name == "model_evaluation" or cfg.pipeline.name == "agg_model_evaluation" or cfg.pipeline.name == "usr_model_evaluation":
        logger.info("Call model evaluation")

        config_file = os.path.join(output_dir, ".hydra", "config.yaml")
        results_folder = os.path.join(output_dir, "results")
        with Runner(f'pipelines/{cfg.pipeline.name}.py', pylint=False).run(log_file=flow_log_file, config_file=config_file, results_folder=results_folder) as running:
            if running.status == 'failed':
                logger.info(f'❌ {running.run} failed:')
            elif running.status == 'successful':
                logger.info(f'✅ {running.run} succeeded:')   
    elif cfg.pipeline.name == "agg_hyperparameter_optimization":     
        logger.info("Call hyperparameter optimization")

        config_file = os.path.join(output_dir, ".hydra", "config.yaml")
        results_folder = os.path.join(output_dir, "results")

        if cfg.pipeline.name[0:3] == "agg":
            search_space_file = os.path.join("config", "model_agg", f"{cfg.model_agg.name}", "search_space.yaml")
        elif cfg.pipeline.name[0:3] == "usr":
            search_space_file = os.path.join("config", "model_usr", f"{cfg.model_usr.name}", "search_space")
        else:
            raise ValueError("cfg.pipeline.name is invalid")
        
        if not os.path.isfile(search_space_file):
            raise FileNotFoundError(f"YAML file not found: {search_space_file}")

        logger.info(f"search_space_file: {search_space_file}")
        shutil.copy2(search_space_file, output_dir)

        with Runner(f'pipelines/{cfg.pipeline.name}.py', pylint=False).run(log_file=flow_log_file, config_file=config_file, results_folder=results_folder, search_space_file=search_space_file) as running:
            if running.status == 'failed':
                logger.info(f'❌ {running.run} failed:')
            elif running.status == 'successful':
                logger.info(f'✅ {running.run} succeeded:')                  
        
    else:
        raise ValueError("error")

    

if __name__ == "__main__":
    run()
