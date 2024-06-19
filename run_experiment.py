from utils import *
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, recall_score, precision_score,balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from model_class import Model
#from model_settings import models_simple, models_behavioral, models_general


def run_experiment_combined(x_train, y_train,  x_test, y_test, name, models, scoring, scaler_object=StandardScaler(), path='', save=True):
    # Get hyperparameter space
    model_info = models[name]
    mod = Model(model_info['model'], name)

    # Fit and save model on training data
    mod.search_and_fit_nested2(x_train, np.array(y_train).ravel(), model_info['param_grid'], scaler_object=scaler_object,
                               scoring=scoring, k=10)

    # Training Scores prep
    cv_results_all = pd.DataFrame(mod.grid_search.cv_results_)

    # Test Scores prep
    y_pred = mod.grid_search.best_estimator_.predict(x_test)
    predict_probs = mod.grid_search.best_estimator_.predict_proba(x_test)

    fpr, tpr, thresholds = roc_curve(y_test,
                                     predict_probs[:, 1],
                                     pos_label=1)

    results = dict(cv_results_all=cv_results_all,
                   training_cv_10=str(list(cv_results_all.loc[mod.grid_search.best_index_, ['split0_test_score',
                                                                                            'split1_test_score',
                                                                                            'split2_test_score',
                                                                                            'split3_test_score',
                                                                                            'split4_test_score',
                                                                                            'split5_test_score',
                                                                                            'split6_test_score',
                                                                                            'split7_test_score',
                                                                                            'split8_test_score',
                                                                                            'split9_test_score']])),
                   training_cv_mean=cv_results_all.loc[mod.grid_search.best_index_, 'mean_test_score'],
                   f1=f1_score(y_test, y_pred),
                   accuracy=accuracy_score(y_test, y_pred),
                   recall=recall_score(y_test, y_pred),
                   precision=precision_score(y_test, y_pred),
                   test_auc=roc_auc_score(y_test, predict_probs[:, 1]),
                   #test_auc=auc(fpr, tpr),
                   test_bacc=balanced_accuracy_score(y_test, y_pred),
                   fpr=fpr,
                   tpr=tpr,
                   thresholds=thresholds,
                   predict_probas=predict_probs[:, 1],
                   predictions=y_pred,
                   true_values=y_test,
                   parameters=mod.grid_search.best_params_)

    logging.info('{} Best Estimator Mean Training Score: {} and test BACC: {} and AUC {}'.format(name,
                                                                                      round(results['training_cv_mean'], 2),
                                                                                      round(results['test_auc'], 2),
                                                                                      round(results['test_bacc'], 2)))

    # Save model
    file = path + '/models/' + 'Finaltest_' + name + '.pkl'
    with open(file, 'wb') as file:
        pickle.dump(mod.grid_search.best_estimator_, file)

    # # TEST Plot and save confusion matrix and ROC
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # labels = ['completed', 'dropped out']
    #
    # ConfusionMatrixDisplay.from_estimator(mod.grid_search, x_test, y_test,
    #                                       display_labels=labels,
    #                                       cmap=plt.cm.Blues, ax=ax)
    # plt.title('Confusion matrix, ' + 'Finaltest_' + '_' + name)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    #
    # # save Confusion Matrix as png
    # fig.savefig(path + '/graphs/' + 'Finaltest_' + '_' + name + '_confusion-matrix_' + '.png')
    # plt.close()
    # del fig
    # # plot ROC curves and save figure
    # fig = plt.figure()
    # lw = 2
    # plt.plot(results['fpr'], results['tpr'], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % results['test_auc'])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Finaltest_ROC ' + name)
    # plt.legend(loc="lower right")
    # fig.savefig(path + '/graphs/' + 'Finaltest_' + name + '_auroc' + '.png', dpi=fig.dpi)
    # # plt.show()
    # plt.close()
    # del fig
    return results
