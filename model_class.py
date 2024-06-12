from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
class Model:
    def __init__(self, model, model_name, parameters=None):
        self.name = model_name
        self.parameters = parameters
        self.model = model

        self.pipeline_all = Pipeline([('scaler', StandardScaler()), (self.name, self.model)])

        if parameters:
            model.set_parameters(**self.parameters)

    def search_and_fit_nested2(self, x_train, y_train, param_grid, scaler_object, scoring,
                               k=10):
        """ Run parameter grid search with k-fold CV
            Input: pre-split data in x and y, parameter grid, optional change in score
            Output: Trained model with the best parameters and respective score
        """

        param_grid = [{self.name + '__' + k: v for k, v in p.items()} for p in param_grid]


        if self.name =='nn':
            self.grid_search = GridSearchCV(self.pipeline_all,
                                        param_grid=param_grid,
                                        n_jobs=1,
                                        cv=k,
                                        scoring=scoring)
        else:
            self.grid_search = GridSearchCV(self.pipeline_all,
                                        param_grid=param_grid,
                                        n_jobs=-1,
                                        cv=k,
                                        scoring=scoring)

        # Small data sets for NN give convergence warnings (as can be expected)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        self.grid_search.fit(x_train, y_train)

        self.best_params = self.grid_search.best_params_
        self.best_score_ = self.grid_search.best_score_
        self.cv_results_ = self.grid_search.cv_results_
