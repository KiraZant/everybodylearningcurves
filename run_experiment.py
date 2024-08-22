from utils.imports import *
import logging
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any
from sklearn.metrics import (
    f1_score, accuracy_score, roc_curve, roc_auc_score,
    recall_score, precision_score, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
from model_class import Model


def run_experiment_combined(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        name: str,
        models: Dict[str, Dict[str, Any]],
        scoring: str,
        scaler_object: Any = StandardScaler(),
        path: str = '',
        save: bool = True
) -> Dict[str, Any]:
    """
    Runs a machine learning experiment by training a model, evaluating it on the test set,
    and saving the results.

    Args:
        x_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        x_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test target data.
        name (str): Name of the model to be used from the models dictionary.
        models (Dict[str, Dict[str, Any]]): Dictionary containing model definitions and hyperparameters.
        scoring (str): Scoring metric to be used for hyperparameter tuning.
        scaler_object (Any): Scaler object to be used for scaling the data. Default is StandardScaler.
        path (str): Directory path where the model should be saved. Default is the current directory.
        save (bool): Flag indicating whether to save the trained model. Default is True.

    Returns:
        Dict[str, Any]: Dictionary containing various metrics and results from the experiment.
    """
    # Get hyperparameter space and initialize the model
    model_info = models[name]
    mod = Model(model_info['model'], name)

    # Fit and save model on training data using nested cross-validation
    mod.search_and_fit_nested2(
        x_train, np.array(y_train).ravel(),
        model_info['param_grid'],
        scaler_object=scaler_object,
        scoring=scoring,
        k=10
    )

    # Prepare training scores from cross-validation results
    cv_results_all = pd.DataFrame(mod.grid_search.cv_results_)

    # Predict on the test set
    y_pred = mod.grid_search.best_estimator_.predict(x_test)
    predict_probs = mod.grid_search.best_estimator_.predict_proba(x_test)

    # Compute ROC curve metrics
    fpr, tpr, thresholds = roc_curve(y_test, predict_probs[:, 1], pos_label=1)

    # Compile results into a dictionary
    results = dict(
        cv_results_all=cv_results_all,
        training_cv_10=str(list(cv_results_all.loc[mod.grid_search.best_index_, [
            'split0_test_score', 'split1_test_score', 'split2_test_score',
            'split3_test_score', 'split4_test_score', 'split5_test_score',
            'split6_test_score', 'split7_test_score', 'split8_test_score',
            'split9_test_score'
            ]])),
        training_cv_mean=cv_results_all.loc[mod.grid_search.best_index_, 'mean_test_score'],
        f1=f1_score(y_test, y_pred),
        accuracy=accuracy_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        test_auc=roc_auc_score(y_test, predict_probs[:, 1]),
        test_bacc=balanced_accuracy_score(y_test, y_pred),
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        predict_probas=predict_probs[:, 1],
        predictions=y_pred,
        true_values=y_test,
        parameters=mod.grid_search.best_params_)

    logging.info('{} Best Estimator Mean Training Score: {} and test BACC: {} and AUC {}'.format(name,
                                                                                                 round(results['training_cv_mean'],2),
                                                                                                 round(results['test_auc'],2),
                                                                                                 round(results['test_bacc'],2)))

    # Save model
    file = path + '/models/' + 'Finaltest_' + name + '.pkl'
    with open(file, 'wb') as file:
        pickle.dump(mod.grid_search.best_estimator_, file)

    return results
