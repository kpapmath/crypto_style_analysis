import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def perform_adf_test(data: pd.Series) -> pd.Series:
    """
    Perform Augmented Dickey-Fuller test to check for stationarity of a time series

    :param data: (pd.Series) Time series data
    :return: (pd.Series) Results of the ADF test
    """
    result = adfuller(data)
    series = pd.Series(result, index=['ADF Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used',
                                    'Critical Values', 'IC Best'])
    series['Check Null Hypothesis'] = 'Stationary' if (series['p-value'] < 0.05) and (abs(series['ADF Test Statistic']) >= abs(series['Critical Values']['5%'])) else 'Non-Stationary'
    return series


def create_adf_result_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame containing the results of the Augmented Dickey-Fuller test for multiple time series

    :param data: (pd.DataFrame) DataFrame containing the time series data
    :return: (pd.DataFrame) DataFrame containing the results of the ADF test for each time series
    """
    adf_results = pd.DataFrame()
    for column in data.columns:
        if ('log_return' in column or 'first_diff' in column):
            adf_results[column] = perform_adf_test(data[column])
        else:
            continue
        adf_results[column] = perform_adf_test(data[column])
    return adf_results


def find_non_stationary_series(data: pd.DataFrame) -> list:
    """
    Find the non-stationary time series in a DataFrame

    :param data: (pd.DataFrame) DataFrame containing the time series data
    :return: (list) list containing the non-stationary time series if all series are stationary the function prints a message
    """
    adf_results = create_adf_result_data_frame(data)
    non_stationary_series = adf_results.loc['Check Null Hypothesis'] == 'Non-Stationary'
    non_stationary_list = [col for col in non_stationary_series.index if non_stationary_series[col]]
    if len(non_stationary_list) == 0:
        print('All series are stationary at 5% significance level')
    else:
        print(f'The following series are non-stationary: {non_stationary_list} at 5% significance level')
        return non_stationary_list


def compute_correlation(data: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """
    Compute the correlation matrix of the time series data.

    :param data: (pd.DataFrame) DataFrame containing the time series data.
    :param column_names: (list) list containing the column names for the heatmap
    :return: (pd.DataFrame) DataFrame containing the correlation matrix.
    """
    df = data.rename(columns={col: column_names[ix] for ix, col in enumerate(data.columns)})
    correlation_matrix = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    return correlation_matrix

def get_column_names_for_heatmap(data: pd.DataFrame) -> list:
    """
    Get the column names for the heatmap

    :param data: (pd.DataFrame) DataFrame containing the time series data
    :return: (list) list containing the column names for the heatmap
    """
    cols = []
    for column in data.columns:
        if (not ('log_return' in column or 'first_diff' in column or 'SOL' in column or 'Date' in column)):
            cols.append(column)
    return cols

if __name__ == '__main__':
    base_dir = os.getenv('BASE_PATH')
    df = pd.read_csv(f'{base_dir}/processed_data_from_2019-01-05_to_2024-07-31.csv', parse_dates=['Date'])
    cols = create_adf_result_data_frame(df).columns
    find_non_stationary_series(df)
    col_new_names = get_column_names_for_heatmap(df)
    compute_correlation(df[cols], col_new_names)
    a = 3