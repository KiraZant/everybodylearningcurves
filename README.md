# everybodylearningcurves

Code used to train the ML models for the paper "Predictive Power, Variance and Generalizability – A Machine Learning Case Study on Minimal Necessary Data Sets Sizes in Mental Health Intervention Predictions".

## File overview

### 🗀 Models
#### 🗎 [model_class.py](./model_class.py)
Utility class for non-sequential scikit-learn models.

#### 🗎 [model_settings.py](./model_settings.py)
Contains all hyperparameters and model settings.

### 🗀 Experiments
#### 🗎 [run_experiment.py](./run_experiment.py)
Calculation of learning curves across data set sizes for one feature group at a time

#### 🗎 [delong_bySunXu.py](./delong_bySunXu.py)
Implementation of deLong test to determine if differences in AUC values are statistically significant


### 🗀 Utils
#### 🗎 [data_prep.py](utils/data_prep.py)
Functions to go from raw data to prepped train-test

#### 🗎 [feature_groups.py](utils/feature_groups.py)
Lists for feature names for the six groups used in the paper

#### 🗎 [logger_code.py](utils/logger_code.py)
Initiates logger to keep track of all characteristics and results

#### 🗎 [make_directory.py](utils/make_directory.py)
Generates result directory for new runs

#### 🗎 [imports.py](utils/imports.py)
Most used packages for easy import

Run by:

	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	python3 main.py

The data used in the study cannot be made public as it's use is restricted per the data user agreement. 
The provided data 'synthetic_data.csv' is entirely synthetic with not resemblence to the real data, therefore, it does not lead to the reported results. 

It was produced via the following code:

X, y = make_classification(n_samples=3654,
                           n_features=len(features),
                           n_redundant=0,
                           n_clusters_per_class=1,
                           random_state=42,
                           weights=[df['dropout_mod3'].value_counts(normalize=True)[0]])

The Jupyter Notebooks including the pre-processing and feature engineering steps are not included as the data cannot be shown for privacy reasons. 
Please refer to Appendix 4 for a detailed description of the feature engineering steps. 

