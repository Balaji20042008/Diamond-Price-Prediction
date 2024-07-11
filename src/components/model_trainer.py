import os, sys

import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class model_trainer_config:
        model_trainer_file_path = os.path.join("artifacts", "model.pkl")

class model_trainer:

    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_training(self, train_arr, test_arr):

        try:

            x_train, y_train, x_test, y_test = (

                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {

                "Linear_Regression": LinearRegression(),
                "Ridge_Regression": Ridge(),
                "Lasso_Regression": Lasso(),
                "Elastic_Net": ElasticNet(),
                "Decision_Tree": DecisionTreeRegressor()
            }

            model_report:dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            print(model_report)

            print("\n=============================================================================\n")

            logging.info(f"Model Report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            print("\n=============================================================================\n")

            logging.info(f"Best Model Found, Model Name: {best_model_name}, R2 Score: {best_model_score}")

            save_object (

                file_path = self.model_trainer_config.model_trainer_file_path,
                obj = best_model
            )

        except Exception as error:

            logging.error("Error in Model Training.")
            raise CustomException(error, sys)
