import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            float
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            float
        """
        try:
            logging.info(f"Calculating Mean Squared Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            return e

class R2Score(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores (R2) for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            f
        """
        try:
            logging.info(f"Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            return e  

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Method to calculate the scores for the model
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            np.array
        """
        try:
            logging.info(f"Calculating Root Mean Squared Error")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            return e 
 