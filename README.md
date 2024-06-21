# everybodylearningcurves

Code used to train the ML models for the paper "Predictive Power, Variance and Generalizability – A Machine Learning Case Study on Minimal Necessary Data Sets Sizes in Mental Health Intervention Predictions".


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

