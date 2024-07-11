import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass

class data_ingestion_config:

    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")
    raw_data_path = os.path.join("artifacts", "raw.csv")

class data_ingestion:

    def __init__ (self):

        self.ingestion_config = data_ingestion_config()

    def initiate_data_congestion (self):

        logging.info("Initiating Data Ingestion")

        try:

            logging.info("Creating a Pandas Dataframe")

            df = pd.read_csv(os.path.join("notebooks/data", "gemstone.csv"))

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("Train-Test Split of the Dataset")

            train_set, test_set = train_test_split(df, test_size = 0.30, random_state = 42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion Completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as error:

            logging.error("Error in Data Ingestion")
            raise CustomException(error, sys)
