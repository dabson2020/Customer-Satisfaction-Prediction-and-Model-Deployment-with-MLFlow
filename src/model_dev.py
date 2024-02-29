import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all model
    """
    @abstractmethod
    def train(self, X_train,y_train) ->None:
        """
        Method to train the model

        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
       """
        
class LinearRegressionModel(Model):
    """
    Class for Linear Regression model
    """
    
    def train(self, X_train, y_train, **kwargs): 

        """
        Method to train the model
        Args:
            X_train: training data
            y_train: training labels
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs) 
            reg.fit(X_train, y_train)
            logging.info(f"Model is trained successfully")   
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            return e