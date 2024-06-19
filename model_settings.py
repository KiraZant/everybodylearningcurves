from utils import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


models_simple = {
    'lr': {
        'model': LogisticRegression(class_weight='balanced'),
        'param_grid': [{'penalty': ['l2', 'l1'],
                        'solver': ['liblinear'],
                        'C': [0.1, 0.5, 1, 5, 10]}
                       ]
             },
    'svm': {
               'model': SVC(probability=True,class_weight='balanced'),
               'param_grid': [{'C': [0.25, 0.5, 1, 2, 3],
                               'kernel': ['linear']},
                              {'C': [0.25, 0.5, 1, 2, 3],
                               'gamma': ['auto'],
                               'kernel': ['rbf']}
                              ]
            },
    'nb': {
            'model': GaussianNB(),
            'param_grid': [{}
                           ]
                 },
    'rf': {
               'model': RandomForestClassifier(class_weight='balanced'),
               'param_grid': [{'n_estimators': [5, 10, 25],
                               'min_samples_split': [10, 25, 50],
                               'max_depth': [25, 50, 100],
                               'bootstrap': [True, False]}
                              ]
           },
    'adaboost': {
                'model': AdaBoostClassifier(algorithm='SAMME'),
                'param_grid': [{'n_estimators':  [25, 50, 100, 250, 500],
                                'learning_rate': [0.5, 0.75, 1]}]
           },
    'nn': {
                'model': MLPClassifier(),
                'param_grid': [{'max_iter': [500],
                                'solver': ['adam'],
                                'hidden_layer_sizes':[(5,), (10,), (15,), (25,), (50,), (75,)]
                                }]
    }
}


models_general = {
    'lr': {
        'model': LogisticRegression(class_weight='balanced'),
        'param_grid': [{'penalty': ['l2', 'l1'],
                        'solver': ['liblinear'],
                        'C': [0.25, 0.5, 1, 5, 10]}
                       ]},
   'svm': {
       'model': SVC(class_weight='balanced', probability=True),
       'param_grid': [{'C': [0.25, 0.5, 1, 2, 3],
                       'kernel': ['linear']},
                      {'C': [0.25, 0.5, 1, 2, 3],
                       'gamma': ['auto'],
                       'kernel': ['rbf']}
                      ]},
    'nb': {
        'model': GaussianNB(),
        'param_grid': [{}]},

   'rf': {
       'model': RandomForestClassifier(class_weight='balanced'),
       'param_grid': [{'n_estimators': [50, 100, 250, 300, 500],
                       'min_samples_split': [25, 50, 75, 100],
                       'max_depth': [50, 100, 250, 500],
                       'bootstrap': [True, False]}
                      ]},
    'adaboost': {
        'model': AdaBoostClassifier(algorithm='SAMME'),
        'param_grid': [{'n_estimators':  [10, 50, 100, 250, 500],
                        'learning_rate': [0.1, 0.5, 0.75, 1, 1.5, 2]}
                       ]},
    'nn': {
        'model': MLPClassifier(),
        'param_grid': [{'hidden_layer_sizes': [(2,), (3,), (10,), (25,), (50,), (100,)],
                       'solver': ['adam']}
                       ]},
}
models_behavioral = {
    'lr': {
        'model': LogisticRegression(class_weight='balanced'),
        'param_grid': [{'penalty': ['l2', 'l1'],
                        'solver': ['liblinear'],
                        'C': [0.01, 0.05, 0.1, 0.5]}
                       ]},
   'svm': {
       'model': SVC(class_weight='balanced', probability=True),
       'param_grid': [{'C': [0.25, 0.5, 1, 2, 3],
                       'gamma': ['auto'],
                       'kernel': ['rbf']}
                      ]},
    'nb': {
        'model': GaussianNB(),
        'param_grid': [{}]},

   'rf': {
       'model': RandomForestClassifier(class_weight='balanced'),
       'param_grid': [{'n_estimators': [250, 300, 500, 750],
                       'min_samples_split': [25, 50, 75, 100],
                       'max_depth': [50, 100, 250, 500, None],
                       'bootstrap': [True, False]}
                      ]},
    'adaboost': {
        'model': AdaBoostClassifier(algorithm='SAMME'),
        'param_grid': [{'n_estimators':  [50, 100, 250, 500],
                        'learning_rate': [0.001, 0.05, 0.1, 0.5, 1, 1.5, 2]}
                       ]},
    'nn': {
        'model': MLPClassifier(),
        'param_grid': [{'hidden_layer_sizes': [(2,), (3,), (25,)],
                       'solver': ['adam']}
                      ]},
}

# Define models and parameter grid per run based on investigating training results
hyper_dict = {'wcs_only': models_simple,
              'baseline_extended': models_general,
              'behavior_simple': models_simple,
              'behavior_extended': models_behavioral,
              'behavior_selected': models_general,
              'all_features': models_general,
              }
