from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from typing import Optional, Dict, Any, List


class Model:
    def __init__(self, model: Any, model_name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Initializes the Model class with the specified model, model name, and parameters.

        Args:
            model (Any): The machine learning model to be used.
            model_name (str): The name of the model, used for identifying steps in a pipeline.
            parameters (Optional[Dict[str, Any]]): Optional hyperparameters to set on the model. Defaults to None.
        """
        self.name = model_name
        self.parameters = parameters
        self.model = model

        # Create a pipeline with scaling and the specified model
        self.pipeline_all = Pipeline([('scaler', StandardScaler()), (self.name, self.model)])

        # Set model parameters if provided
        if parameters:
            model.set_params(**self.parameters)

    def search_and_fit_nested2(
            self,
            x_train: Any,
            y_train: Any,
            param_grid: List[Dict[str, Any]],
            scaler_object: Any,
            scoring: str,
            k: int = 10
    ) -> None:
        """
        Runs a parameter grid search with k-fold cross-validation and fits the model with the best parameters.

        Args:
            x_train (Any): Training data features.
            y_train (Any): Training data labels.
            param_grid (List[Dict[str, Any]]): List of dictionaries specifying the parameter grid for the search.
            scaler_object (Any): Scaler object to be used in the pipeline (though not explicitly used here).
            scoring (str): The scoring method to evaluate the model.
            k (int): Number of folds for cross-validation. Defaults to 10.

        Returns:
            None
        """
        # Adjust parameter grid to match pipeline naming convention
        param_grid = [{self.name + '__' + k: v for k, v in p.items()} for p in param_grid]

        # Configure GridSearchCV with different settings based on the model type
        if self.name == 'nn':
            self.grid_search = GridSearchCV(
                self.pipeline_all,
                param_grid=param_grid,
                n_jobs=1,  # Use single CPU core for neural networks
                cv=k,
                scoring=scoring
            )
        else:
            self.grid_search = GridSearchCV(
                self.pipeline_all,
                param_grid=param_grid,
                n_jobs=-1,  # Use all available CPU cores
                cv=k,
                scoring=scoring
            )

        # Suppress convergence warnings for models prone to such issues (e.g., neural networks)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        # Fit the model using the grid search
        self.grid_search.fit(x_train, y_train)

        # Store the best parameters, score, and cross-validation results
        self.best_params = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        self.cv_results_ = self.grid_search.cv_results_
