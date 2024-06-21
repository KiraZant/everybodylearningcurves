from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
#from feature_groups import basic_info, behavior_simple, wcs_scores, ext_quest, behavior, behavior_all

def split(df, Xs, ys='dropout_mod3', sampling_N=False, state_sample=1, test_state=40, save=False):
    '''
    Function bases on pre-processed dataset and splits in  train/test sets + samples testset
    :param df: Pre-processed dataset
    :param sampling_N: Integer for number of samples in train, test stays the same
    :param ys: What will be predicted, assumed to be dropout (could also be Outcome)
    :return: y-stratified and sampled training and full test data
    '''

    x = df[Xs+[ys]].copy(deep=True)
    y = df[[ys]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=test_state)

    x_test = x_test.drop(columns=[ys])

    logging.info('Distribution {}'.format(y_train.value_counts().to_string()))

    if sampling_N < len(x_train):
        x_train = x_train.groupby(ys, group_keys=False).apply(lambda x: x.sample(frac=sampling_N/len(x_train), random_state=state_sample))

    x_train = x_train.drop(columns=[ys])
    y_train = y_train.loc[[ID for ID in x_train.index]]

    logging.info('Distribution AFTER sampling {}'.format(y_train.value_counts().to_string()))

    if save:
        ignore_col = [col for col in df.columns if col not in ['remission',
                                                               'total_sessions', 'dropout_mod4', 'study_entry',
                                                               'study_arm',
                                                               'dropout_mod3r', 'age.1', 'holdout',
                                                               'client_database_id']]
        df['holdout'] = np.where(df.index.isin(x_test.index), 1, 0)
        means = pd.concat(
            [df[ignore_col].mean(), df[ignore_col].std(), df[ignore_col].min(), df[ignore_col].max(), df[ignore_col + ['holdout']].groupby(by='holdout').mean().T],
            axis=1)
        means.columns = ['Total Mean', 'Total STD', 'Total Min.', 'Total Max.', 'Training Mean', 'Test Mean']
        means.to_csv('/Users/kirstenzantvoort/PycharmProjects/Icare_mlcurves/Results/seed40_auc/mean_all_train_test.csv')

    return x_train, x_test, y_train, y_test

def cut_predtime(df_1, pred_cut, date_ditct, date_df_1):
    df_1['start_date'] = df_1['client_database_id'].map(date_ditct)
    df_1.start_date = pd.to_datetime(df_1.start_date)
    df_1['intervention_day'] = pd.to_datetime(df_1[date_df_1].dt.date)-df_1['start_date']
    df_1['intervention_day'] = df_1['intervention_day'].dt.days
    df_1['intervention_week'] = np.where(df_1['intervention_day'] < 7, 'week1',
                                np.where(df_1['intervention_day'] < 14, 'week2',
                                np.where(df_1['intervention_day'] < 21, 'week3',
                                np.where(df_1['intervention_day'] < 28, 'week4', None))))
    # Cut all messages sent after the first four weeks
    print('DataFrame rows before filtering', len(df_1))
    df_1 = df_1[df_1['intervention_day'] < pred_cut]
    print('DataFrame rows AFTER filtering',len(df_1))
    return df_1

def add_startdate(df_1, date_ditct, date_df_1):
    df_1['start_date'] = df_1['client_database_id'].map(date_ditct)
    df_1.start_date = pd.to_datetime(df_1.start_date)
    df_1['intervention_day'] = (pd.to_datetime(df_1[date_df_1])-pd.to_datetime(df_1['start_date'])).dt.days
    df_1['intervention_week'] = np.where(df_1['intervention_day'] < 7, 1,
                                np.where(df_1['intervention_day'] < 14, 2,
                                np.where(df_1['intervention_day'] < 21, 3,
                                np.where(df_1['intervention_day'] < 28, 4,
                                np.where(df_1['intervention_day'] < 35, 5,
                                np.where(df_1['intervention_day'] < 42, 6,
                                np.where(df_1['intervention_day'] < 49, 7,
                                np.where(df_1['intervention_day'] < 56, 8,
                                np.where(df_1['intervention_day'] < 63, 9,
                                np.where(df_1['intervention_day'] < 70, 10,
                                np.where(df_1['intervention_day'] < 77, 11,
                                np.where(df_1['intervention_day'] < 84, 12, None))))))))))))
    return df_1

def imput_mis(df, relevant_col):
    x_train, x_test, y_train, y_test = split(df, relevant_col)
    df['holdout'] = np.where(df.user_id.isin(x_train.index), 0, 1)
    df = df.drop(columns=relevant_col)

    # Copy the data
    x_train_imp = x_train[relevant_col].copy(deep=True)
    x_test_imp = x_test[relevant_col].copy(deep=True)

    # Init imputer
    ii_imp = IterativeImputer(
        estimator=ExtraTreesRegressor(), max_iter=10, random_state=1121218)

    # Fit + transform training set
    x_train_imp.loc[:, :] = ii_imp.fit_transform(x_train_imp)

    # Use imputer to transform test
    x_test_imp.loc[:, :] = ii_imp.transform(x_test_imp)

    df = df.merge(x_train_imp, left_index=True, right_index=True, how='left')
    df = x_test_imp.combine_first(df)

    return df