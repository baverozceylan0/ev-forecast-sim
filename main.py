import os
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

    
    logger = setup_logger("main", log_file=main_log_file) 
    logger.info("Logging is working.")


    if cfg.pipeline.name == "model_evaluation":
        logger.info("Call model evaluation")

        config_file = os.path.join(output_dir, ".hydra", "config.yaml")
        results_folder = os.path.join(output_dir, "results")
        with Runner('pipelines/model_evaluation.py', pylint=False).run(log_file=flow_log_file, config_file=config_file, results_folder=results_folder) as running:
            if running.status == 'failed':
                logger.info(f'❌ {running.run} failed:')
            elif running.status == 'successful':
                logger.info(f'✅ {running.run} succeeded:')                
            logger.info(f'[metaflow]\n ---------- stderr ----------\n{running.stderr} ----------------------------\n')
            logger.info(f'[metaflow]\n ---------- stdout ----------\n{running.stdout} ----------------------------\n')
        
    else:
        raise ValueError("error")

    

if __name__ == "__main__":
    run()
