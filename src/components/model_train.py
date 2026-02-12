import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRFRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        logging.info("Starting model training with hyperparameter tuning")

        try:
            # Split features and target
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Models dictionary
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRFRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            # Hyperparameter grids (skip tuning for CatBoost later)
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                },
                "Linear Regression": {},
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6]
                },
                "CatBoost": {}  # Skip RandomizedSearch for compatibility
            }

            best_model = None
            best_score = float("-inf")
            best_model_name = None

            # Training Loop
            for model_name, model in models.items():

                logging.info(f"Training model: {model_name}")

                param = params[model_name]

                # Skip RandomizedSearch for CatBoost (compatibility fix)
                if model_name == "CatBoost":
                    model.fit(X_train, y_train)
                    trained_model = model

                elif param:
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param,
                        n_iter=3,
                        cv=3,
                        scoring="r2",
                        random_state=42,
                        n_jobs=-1
                    )

                    random_search.fit(X_train, y_train)
                    trained_model = random_search.best_estimator_

                else:
                    model.fit(X_train, y_train)
                    trained_model = model

                # Evaluate
                y_test_pred = trained_model.predict(X_test)
                test_score = r2_score(y_test, y_test_pred)

                logging.info(f"{model_name} Test R2 Score: {test_score}")

                # Track Best Model
                if test_score > best_score:
                    best_score = test_score
                    best_model = trained_model
                    best_model_name = model_name

            print(f"Best Model After Tuning: {best_model_name}")
            print(f"Best R2 Score: {best_score}")
   

            logging.info(
                f"Best model selected: {best_model_name} with R2 score {best_score}"
            )

            if best_score < 0.6:
                raise CustomException("No model achieved acceptable performance", sys)

            # Save Best Model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_score

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)
