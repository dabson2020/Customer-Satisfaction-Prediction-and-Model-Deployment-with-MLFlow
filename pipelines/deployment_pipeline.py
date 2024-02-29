import numpy as np
import pandas as pd
#from materializer.custom_materializer import cs_materializer
from steps.clean_data import clean_data
from steps.evaluate import evaluate_model
from steps.ingest_data import ingest_df
from steps.train_model import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger configuration"""
    min_accuracy: float = 0.92

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
   
):
    """Implements a simple model that looks at the input model 
    accuracy and determines whether the model should be deployed or not."""
    return accuracy >= config.min_accuracy

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path="/Users/adeol/Desktop/MLFlow-Project/data/olist_customers_dataset.csv")
    X_train, X_test, y_train, y_test = clean_data(data=df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, mse, rmse = evaluate_model(model, X_test, y_test)
    deployer_decision = deployment_trigger(rmse)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployer_decision,
        workers=workers,
        timeout=timeout,

    )



