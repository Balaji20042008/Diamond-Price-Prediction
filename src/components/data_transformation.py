from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

import pandas as pd
import numpy as np

@dataclass
class data_transformation_config:
    def __init__ (self):
        self.preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class data_transformation:
    def __init__ (self):
        self.data_transformation_config = data_transformation_config()
    
    def get_data_transformation_object(self):

        try:

            logging.info("Initiating Data Transformation")

            numerical_features = ["carat", "depth", "table", "x", "y", "z"]
            categorical_fetaures = ["cut", "color", "clarity"]

            cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            logging.info("Initiating Pipeline")

            numerical_pipeline = Pipeline(steps = [
                ("Imputer", SimpleImputer(strategy = "median")),
                ("Scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps = [
                ("Imputer", SimpleImputer(strategy = "most_frequent")),
                ("Encoder", OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories])),
                ("Scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers = [
                ("Numerical_Pipeline", numerical_pipeline, numerical_features),
                ("Categorical_Pipeline", categorical_pipeline, categorical_fetaures)
            ])

            logging.info("Completed Pipeline")

            return preprocessor

        except Exception as error:

            logging.error("Error in Data Transformation Object")
            raise CustomException(error, sys)
    
    def intiate_data_transformation(self, train_path, test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train & Test Data")
            logging.info(f"train_df.head(): \n{train_df.head().to_string()}")
            logging.info(f"test_df.head(): \n{test_df.head().to_string()}")

            logging.info("Obtain Preprocessor Object")

            preprocessor_obj = self.get_data_transformation_object()

            target_column = "price"
            drop_columns = [target_column, "id"]

            input_feature_train_df = train_df.drop(columns = drop_columns, axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = drop_columns, axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("Preprocessing on Train & Test Datasets")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object (

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info("Creation of Processor Pickle File")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as error:

            logging.error("Error in Data Transformation Initiation")
            raise CustomException(error, sys)


