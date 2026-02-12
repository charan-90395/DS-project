import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.

    Parameters:
    file_path (str): Path where object will be saved.
    obj: Python object to save.

    Raises:
    CustomException: If error occurs during saving.
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object successfully saved at {file_path}")

    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple ML models and return performance report.

    Parameters:
    X_train : Training features
    y_train : Training labels
    X_test  : Testing features
    y_test  : Testing labels
    models  : Dictionary of model name and model object

    Returns:
    dict : Dictionary containing model name and test R2 score

    Raises:
    CustomException: If error occurs during evaluation
    """
    try:
        model_report = {}

        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store test score
            model_report[model_name] = test_model_score

            logging.info(
                f"{model_name} - Train R2: {train_model_score}, Test R2: {test_model_score}"
            )

        return model_report

    except Exception as e:
        logging.error(f"Error evaluating models: {e}")
        raise CustomException(e, sys)
