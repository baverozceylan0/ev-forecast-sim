python main.py pipeline.name="agg_model_evaluation" model_agg=kNN/champion 

python main.py pipeline.name="agg_model_evaluation" model_agg=XGBRegressor/default_parameters 

python main.py pipeline.name="agg_model_evaluation" model_agg=XGBRegressor/default model_agg.name="XGBRegressor" model_agg.version="tmp" 

python main.py pipeline.name="agg_hyperparameter_optimization" model_agg=XGBRegressor/default_parameters

python main.py pipeline.name="agg_hyperparameter_optimization" model_agg=XGBRegressor/default_parameters