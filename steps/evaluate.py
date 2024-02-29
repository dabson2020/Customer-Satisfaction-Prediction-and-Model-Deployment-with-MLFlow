import logging

import mlflow
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import R2Score,MSE,RMSE
from typing_extensions import Annotated
from typing import Tuple

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin,
                   X_test:pd.DataFrame,
                   y_test: pd.Series) -> Tuple[
                       Annotated[float, 'mse'], 
                       Annotated[float, 'r2'], 
                       Annotated[float, 'rmse'],
                       ]:
    """
    Method to evaluate the model on test data
    Args:
        df: the ingested data

    """
    try:

        prediction = model.predict(X_test)
        

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE", mse)

        r2_class = R2Score()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("RMSE", rmse)

        return mse, r2, rmse
    except Exception as e:  
        logging.error(f"Error in evaluating model: {e}")
        return e