print("MODEL_TRAIN FILE LOADED")

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from xgboost import XGBRFRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered the model trainer method or component")
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], #take all rows and col except last col "X"
                train_array[:, -1], #take all rows and take only the last col "Y"
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRFRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"\nBest Model: {best_model_name}")
            print(f"Best R2 Score: {best_model_score}\n")
            
            logging.info("Best model found, model name: %s, R2 score: %s", best_model_name, best_model_score)

            if best_model_score < 0.6:
                logging.info("No best model found with R2 score above 0.6")
                raise CustomException("No best model found", sys)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            logging.error("Error occurred during model training: %s", str(e))
            raise CustomException(e, sys)

            
