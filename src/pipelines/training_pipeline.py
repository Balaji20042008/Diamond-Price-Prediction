import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import data_ingestion
from src.components.data_transformation import data_transformation
from src.components.model_trainer import model_trainer

if __name__ == "__main__":

    data_ingestion_obj = data_ingestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_congestion()
    print("Train Data Path: ", train_data_path)
    print("Test Data Path: ", test_data_path)

    data_transformation_obj = data_transformation()
    train_arr, test_arr, _ = data_transformation_obj.intiate_data_transformation(train_data_path, test_data_path)

    model_trainer_obj = model_trainer()
    model_trainer_obj.initiate_model_training(train_arr,test_arr)