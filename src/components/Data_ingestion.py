# “In data ingestion, we collect data from multiple sources, 
# validate its schema and quality, handle basic transformations, log the process, a
# nd store the cleaned raw data for downstream processing.”

import os
import sys
#Used to get system-level error details and Helpful for custom exception handling :sys

from src.exception import CustomException #wraps Python errors with extra info (file name, line number)
from src.logger import logging   #logs messages into a log file instead of using print()

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_train import ModelTrainer #To use the model trainer config class for file paths
from src.components.model_train import ModelTrainerConfig #To use the model trainer config class for file paths
@dataclass
class DataIngestionConfig:  #This class only stores paths.
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
class DataIngestion:  #This is the actual ingestion logic class.
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #Creates an instance of the config class to access file paths.

    def initiate_data_ingestion(self):   #This method performs the data ingestion steps: reading, saving raw, splitting, and saving train/test.
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'ANL-FOR-EDA\stud.csv')
            logging.info("Read the dataset as a dataframe from path: %s", 'ANL-FOR-EDA\stud.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) #exit_ok=avoids error if folder already exists

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            #In df.to_csv(), the actual data is stored inside the DataFrame df in memory, 
            # and to_csv() writes that data to a file at the specified path.
            logging.info("Saved the raw data to path: %s", self.ingestion_config.raw_data_path)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Split the data into train and test sets and saved to paths: %s and %s",
                         self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)  
#Returns the file paths of the train and test data for downstream use (like in data transformation or model training).
        except Exception as e:
            logging.error("Error occurred during data ingestion: %s", str(e))
            raise CustomException(e, sys)
        
if __name__ == "__main__": #This block allows us to run this script directly for testing the data ingestion process.

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)

