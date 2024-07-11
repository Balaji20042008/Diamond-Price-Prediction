import os, sys, pickle
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as error:
        
        raise CustomException(error, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models):

    try:

        report = {}
        
        for i in range(len(models)):

            model = list(models.values())[i]
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)

            model_score = r2_score(y_test, y_pred)

            report[list(models.keys())[i]] = model_score

        return report
    
    except Exception as error:

        raise CustomException(error, sys)
    
def load_object(file_path):

    try:

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj) 

    except Exception as error:

        raise CustomException(error, sys)