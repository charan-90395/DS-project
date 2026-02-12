import numpy as np
import dill
import pandas as pd
import os
import sys

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    This function saves a Python object to a file using dill.
    
    Parameters:
    file_path (str): The path where the object will be saved.
    obj: The Python object to be saved.
    
    Raises:
    CustomException: If there is an error during the saving process.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file path
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info(f"Object successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)
