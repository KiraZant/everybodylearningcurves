from utils.imports import *
from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def split(
    df: pd.DataFrame,
    Xs: list,
    ys: str = 'dropout_mod3',
    sampling_N: int = False,
    state_sample: int = 1,
    test_state: int = 40,
    save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the pre-processed dataset into train/test sets and optionally samples the training set.

    Args:
        df (pd.DataFrame): The pre-processed dataset.
        Xs (list): List of feature columns to be used for prediction.
        ys (str): The target variable to be predicted. Default is 'dropout_mod3'.
        sampling_N (int): Number of samples in the training set. Test set size remains unchanged. Default is False.
        state_sample (int): Random state for sampling. Default is 1.
        test_state (int): Random state for train/test splitting. Default is 40.
        save (bool): Flag to save the train/test split statistics. Default is False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x_train, x_test, y_train, y_test dataframes after the split and sampling.
    """
    # Create a copy of the relevant columns
    x = df[Xs + [ys]].copy(deep=True)
    y = df[[ys]]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=test_state)

    # Drop the target variable from the test set features
    x_test = x_test.drop(columns=[ys])

    # Log the distribution of the target variable in the training set
    logging.info('Distribution {}'.format(y_train.value_counts().to_string()))

    # Sample the training set if sampling_N is provided and less than the size of the training set
    if sampling_N < len(x_train):
        x_train = x_train.groupby(ys, group_keys=False).apply(
            lambda x: x.sample(frac=sampling_N/len(x_train), random_state=state_sample)
        )

    # Update y_train to include only the sampled instances
    x_train = x_train.drop(columns=[ys])
    y_train = y_train.loc[x_train.index]

    # Log the distribution of the target variable after sampling
    logging.info('Distribution AFTER sampling {}'.format(y_train.value_counts().to_string()))

    # Save the train/test split statistics if save flag is set
    if save:
        ignore_col = [col for col in df.columns if col not in [
            'remission', 'total_sessions', 'dropout_mod4', 'study_entry',
            'study_arm', 'dropout_mod3r', 'age.1', 'holdout', 'client_database_id'
        ]]
        df['holdout'] = np.where(df.index.isin(x_test.index), 1, 0)
        means = pd.concat(
            [
                df[ignore_col].mean(), df[ignore_col].std(),
                df[ignore_col].min(), df[ignore_col].max(),
                df[ignore_col + ['holdout']].groupby(by='holdout').mean().T
            ], axis=1
        )
        means.columns = ['Total Mean', 'Total STD', 'Total Min.', 'Total Max.', 'Training Mean', 'Test Mean']
        means.to_csv('/Users/kirstenzantvoort/PycharmProjects/Icare_mlcurves/Results/seed40_auc/mean_all_train_test.csv')

    return x_train, x_test, y_train, y_test

def cut_predtime(
    df_1: pd.DataFrame,
    pred_cut: int,
    date_ditct: Dict[Any, Any],
    date_df_1: str
) -> pd.DataFrame:
    """
    Filters the DataFrame to include only records before a specific prediction cutoff time.

    Args:
        df_1 (pd.DataFrame): The DataFrame containing intervention data.
        pred_cut (int): The cutoff time (in days) for predictions.
        date_ditct (Dict[Any, Any]): Dictionary mapping client IDs to their start dates.
        date_df_1 (str): Column name in df_1 containing dates for the intervention.

    Returns:
        pd.DataFrame: The filtered DataFrame with only records before the cutoff time.
    """
    # Map client start dates and calculate intervention days
    df_1['start_date'] = df_1['client_database_id'].map(date_ditct)
    df_1.start_date = pd.to_datetime(df_1.start_date)
    df_1['intervention_day'] = pd.to_datetime(df_1[date_df_1].dt.date) - df_1['start_date']
    df_1['intervention_day'] = df_1['intervention_day'].dt.days

    # Categorize the intervention day into weeks
    df_1['intervention_week'] = np.where(
        df_1['intervention_day'] < 7, 'week1',
        np.where(df_1['intervention_day'] < 14, 'week2',
        np.where(df_1['intervention_day'] < 21, 'week3',
        np.where(df_1['intervention_day'] < 28, 'week4', None)))
    )

    # Print DataFrame size before filtering
    print('DataFrame rows before filtering', len(df_1))

    # Filter the DataFrame based on the prediction cutoff time
    df_1 = df_1[df_1['intervention_day'] < pred_cut]

    # Print DataFrame size after filtering
    print('DataFrame rows AFTER filtering', len(df_1))

    return df_1

def add_startdate(
        df_1: pd.DataFrame,
    date_ditct: Dict[Any, Any],
    date_df_1: str) -> pd.DataFrame:
    """
    Adds start dates and calculates the intervention days and weeks for each record in the DataFrame.

    Args:
        df_1 (pd.DataFrame): The DataFrame containing intervention data.
        date_ditct (Dict[Any, Any]): Dictionary mapping client IDs to their start dates.
        date_df_1 (str): Column name in df_1 containing dates for the intervention.

    Returns:
        pd.DataFrame: The DataFrame with added start date, intervention day, and intervention week.
    """
    # Map client start dates and calculate intervention days
    df_1['start_date'] = df_1['client_database_id'].map(date_ditct)
    df_1.start_date = pd.to_datetime(df_1.start_date)
    df_1['intervention_day'] = (pd.to_datetime(df_1[date_df_1]) - pd.to_datetime(df_1['start_date'])).dt.days

    # Categorize the intervention day into weeks
    df_1['intervention_week'] = np.where(
        df_1['intervention_day'] < 7, 1,
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

def imput_mis(df: pd.DataFrame, relevant_col: list) -> pd.DataFrame:
    """
    Imputes missing values in the relevant columns of the DataFrame using IterativeImputer.

    Args:
        df (pd.DataFrame): The DataFrame containing the data with missing values.
        relevant_col (list): List of columns to perform imputation on.

    Returns:
        pd.DataFrame: The DataFrame with imputed values.
    """
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = split(df, relevant_col)
    df['holdout'] = np.where(df.user_id.isin(x_train.index), 0, 1)
    df = df.drop(columns=relevant_col)

    # Copy the relevant columns for imputation
    x_train_imp = x_train[relevant_col].copy(deep=True)
    x_test_imp = x_test[relevant_col].copy(deep=True)

    # Initialize the IterativeImputer with ExtraTreesRegressor
    ii_imp = IterativeImputer(
        estimator=ExtraTreesRegressor(), max_iter=10, random_state=1121218
    )

    # Fit and transform the training set
    x_train_imp.loc[:, :] = ii_imp.fit_transform(x_train_imp)

    # Transform the test set
    x_test_imp.loc[:, :] = ii_imp.transform(x_test_imp)

    # Merge the imputed training data back into the original DataFrame
    df = df.merge(x_train_imp, left_index=True, right_index=True, how='left')

    # Combine the imputed test data with the DataFrame, prioritizing test data values
    df = x_test_imp.combine_first(df)

    return df