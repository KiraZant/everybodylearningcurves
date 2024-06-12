from utils import *
from data_prep import split
from feature_groups import *
from model_settings import *
from logger_code import initiate_log
from save_and_pull import generate_path
from run_experiment import run_experiment_combined
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    PATH = './'
    logger = initiate_log(PATH)

    # Step 1: Check Data Version
    data_version = ''
    logging.info('Data Version - {}'.format(data_version))
    df = pd.read_csv(PATH + data_version).set_index('user_id')

    # Step 2: Define what variables to use (keep order)
    Xs = wcs_scores + intr_set + socio_dem + psych_history + personality + expectation + other_psyc
    # #behavior_simple + wcs_scores + intr_set + socio_dem + psych_history + personality + expectation + other_psyc + behavior + behavior_all

    Xs = Xs+basic_info

    # Step 3: Give this run a name
    run = 'baseline_extended'

    # Step 4: Decide for which Ns the run is completing
    all_ns = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3651]#[100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3651]

    # Take away the simulated 20% test set
    all_ns = [round(i*0.8) for i in all_ns]

    # Step 5: Define number of repetitions for small data and where to start the training data seed
    reps_small = 10
    reps_medium = 5
    reps_large = 3

    # Step 6: Define scoring to optimize models on
    scoring = 'roc_auc'

    # Step 7: Choose model types and pull hyperparamaters for it
    model_considered = ['lr', 'svm', 'nb', 'rf', 'adaboost', 'nn']

    try:
        # pulled from model settings
        models = hyper_dict[run]
    except:
        models = hyper_dict['trial']
        print('No specified hyperparameters set')
    models = {key: models[key] for key in model_considered}

    path = './Results/' + run + '/'

    for i in all_ns:
        if i <= 500:
            reps = reps_small
        elif i >= 2000:
            reps = reps_large
        else:
            reps = reps_medium

        while reps > 0:
            if os.path.exists(path+'allresults.csv'):
                result_curves = pd.read_csv(path+'allresults.csv')
                j = result_curves.index.max()+1

            else:
                result_curves = pd.DataFrame(index=range(0, len(all_ns)), columns=['N', 'cvs_'+scoring, 'model', 'test_auc',
                                                                                   'test_bacc', 'hyperparameters',
                                                                                   'y_pred', 'y_pred_prob'])
                j = 0

            logging.info('Start with'+str(i)+' with'+str(reps)+' runs to go')
            x_train, x_test, y_train, y_test = split(df, Xs, sampling_N=i, state_sample=42+reps)

            name = 'Run_' + str(j) + '_' + str(i)

            generate_path(path+name)
            logging.info('Features included {}'.format(str(x_train.columns)))

            for model in model_considered:
                results = run_experiment_combined(x_train, y_train, x_test, y_test,
                                                      name=model,
                                                      scaler_object=StandardScaler(),
                                                      scoring=scoring,
                                                      path=path + name,
                                                      models=models)

                result_curves.loc[j, 'N'] = i
                result_curves.loc[j, 'cvs_'+scoring] = results['training_cv_10']
                result_curves.loc[j, 'model'] = model

                result_curves.loc[j, 'test_auc'] = results['test_auc']
                result_curves.loc[j, 'test_bacc'] = results['test_bacc']
                result_curves.loc[j, 'hyperparameters'] = str(results['parameters'])
                result_curves.loc[j, 'y_pred'] = str(results['predictions'])
                result_curves.loc[j, 'y_pred_prob'] = str(results['predict_probas'])

                j += 1
            result_curves.to_csv(path+'/allresults.csv', index=None)
            reps = reps-1

    logging.info('Finished')