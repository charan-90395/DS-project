import sys                                  # Used for exception handling
from dataclasses import dataclass            # Used to create config class

import numpy as np                           # For numerical operations
import pandas as pd                          # For reading CSV files

# Scikit-learn preprocessing tools
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project-specific utilities
from src.exception import CustomException    # Custom exception class
from src.logger import logging               # Logging utility
import os                                   
from src.utils import save_object            # Function to save objects

from src.utils import save_object

# ===================== CONFIGURATION CLASS =====================

@dataclass
class DataTransformationConfig:
    # Path where the trained preprocessing object will be saved
    preprocessor_obj_file_path = os.path.join(
        "artifacts", "proprocessor.pkl"
    )


# ===================== DATA TRANSFORMATION CLASS =====================

class DataTransformation:
    def __init__(self):
        # Initialize configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This method creates and returns the preprocessing object
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "math_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Numerical pipeline: handle missing values + scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline: handle missing values + encoding + scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Log column information
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            # Raise custom exception if any error occurs
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This method reads data, applies preprocessing,
        saves the preprocessing object, and returns transformed data
        """
        try:
            # Read training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object() 
            """Calls the earlier function

            Gets the pipeline that knows how to:

            fill missing values

            encode text

            scale numbers
            """

            # Define target column
            target_column_name = "reading_score"

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training and testing data"
            )

            # Fit preprocessing on training data and transform both datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Combine features and target into single arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            # Save preprocessing object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed data and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            # Handle errors using custom exception
            raise CustomException(e, sys)
