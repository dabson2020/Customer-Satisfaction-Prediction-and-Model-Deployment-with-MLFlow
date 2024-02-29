from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from steps.ingest_data import ingest_df
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate import evaluate_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(data_path: str = "/home/adeola/mlproject/Customer-Satisfaction-Prediction-and-Model-Deployment-with-MLFlow/data/olist_customers_dataset.csv"):
    """
    The training pipeline
    Args:
        data_path: path to the data
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test =clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2, mse, rmse =evaluate_model(model, X_test, y_test)