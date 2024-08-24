import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(find_dotenv())


def create_dir() -> str:
    """
    This function creates a directory to save the data if it does not exist
    :return: path to the directory
    """
    current_dir = os.getcwd()
    # current_dir = str(Path(current_dir).parents[0])
    path = os.path.join(current_dir, 'data')
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_data(file_path: str,
              date_col: str = 'Date',
              sep=',',
              from_date: str = '2019-01-04',
              to_date: str = '2024-07-31'
              ) -> pd.DataFrame:
    """
    This function loads the data from a csv file and filters the data based on the date
    :param file_path: the path to the file
    :param date_col: the column containing the date
    :param sep: the separator used in the csv file
    :param from_date: date to start receiving the data (format: 'YYYY-MM-DD')
    :param to_date: date to end receiving the data (format: 'YYYY-MM-DD')
    :return: dataframe containing the data form date to date
    """
    df = pd.read_csv(file_path, sep=sep, parse_dates=[date_col], date_format='mixed')
    return df[(df[date_col] >= from_date) & (df[date_col] <= to_date)]


def convert_to_datetime(df_import: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function converts the column to datetime
    :param df_import: the dataframe to convert
    :param col_name: the column to convert
    :return: the dataframe with the column converted to datetime
    """
    df_import[col_name] = pd.to_datetime(df_import[col_name], format='%Y-%m-%d')
    return df_import


def preprocess_yield_data(path: str = 'data/',
                          from_date: str = '2019-01-04',
                          to_date: str = '2024-07-31',
                          fred_data_1: str = 'fred_data_from_2019-01-01_to_2024-08-12_DGS1.csv',
                          fred_data_10: str = 'fred_data_from_2019-01-01_to_2024-08-12_DGS10.csv',
                          japan_data: str = 'jgbcme_all.csv',
                          china_data: str = 'china_yield.csv',
                          ecb_1_data: str = 'ECB Data Portal_20240812111256.csv',
                          ecb_10_data: str = 'ECB Data Portal_20240812111349.csv'
                          ) -> pd.DataFrame:
    """
    This function preprocesses the yield data
    :param path: the path to the data
    :param from_date: date to start receiving the data (format: 'YYYY-MM-DD')
    :param to_date: date to end receiving the data (format: 'YYYY-MM-DD')
    :param fred_data_1: the file containing the 1Y yield data
    :param fred_data_10: the file containing the 10Y yield data
    :param japan_data: the file containing the Japan yield data
    :param china_data: the file containing the China yield data
    :param ecb_1_data: the file containing the ECB 1Y yield data
    :param ecb_10_data: the file containing the ECB 10Y yield data
    :return: dataframe containing the yield data for 1Y and 10Y for US, Japan, China, and EU
    """
    file_path = path + fred_data_1
    date_col = 'Date'
    fred_1 = load_data(file_path, date_col)
    fred_1.rename(columns={'value': '1Y_US'}, inplace=True)
    file_path = path + fred_data_10
    fred_10 = load_data(file_path, date_col, from_date=from_date, to_date=to_date)
    fred_10.rename(columns={'value': '10Y_US'}, inplace=True)
    fred = pd.merge_ordered(fred_1, fred_10, on='Date')
    if 'DGS1' in fred.columns:
        fred.rename(columns={'DGS1': '1Y_US'}, inplace=True)
    if 'DGS10' in fred.columns:
        fred.rename(columns={'DGS10': '10Y_US'}, inplace=True)
    file_path = path + japan_data
    sep = ','
    jgb = pd.read_csv(file_path, sep=sep, header=[0, 1])
    jgb.columns = jgb.columns.droplevel(0)
    jgb['Date'] = pd.to_datetime(jgb['Date'], format='%Y/%m/%d')
    jgb[['1Y', '10Y']] = jgb[['1Y', '10Y']].apply(pd.to_numeric, errors='coerce')
    jgb = jgb[['Date', '1Y', '10Y']][(jgb['Date'] >= '2019-01-04') & (jgb['Date'] <= '2024-07-31')]
    jgb.rename(columns={'1Y': '1Y_JP', '10Y': '10Y_JP'}, inplace=True)
    df_temp = pd.merge_ordered(fred, jgb, on='Date')
    file_path = path + china_data
    date_col = 'Date'
    sep = ';'
    china = load_data(file_path, date_col, sep, from_date=from_date, to_date=to_date)
    china = china[['Date', '1Y', '10Y']]
    df_temp = pd.merge_ordered(df_temp, china, on='Date')
    df_temp.rename(columns={'1Y': '1Y_CH', '10Y': '10Y_CH'}, inplace=True)
    file_path = path + ecb_1_data
    date_col = 'DATE'
    sep = ','
    ecb_1 = load_data(file_path, date_col, sep, from_date=from_date, to_date=to_date)
    ecb_1 = ecb_1.iloc[:, [0, 2]]
    ecb_1.rename(columns={ecb_1.columns[0]: 'Date', ecb_1.columns[1]: '1Y'}, inplace=True)
    file_path = path + ecb_10_data
    date_col = 'DATE'
    sep = ','
    ecb_10 = load_data(file_path, date_col, sep, from_date=from_date, to_date=to_date)
    ecb_10 = ecb_10.iloc[:, [0, 2]]
    ecb_10.rename(columns={ecb_10.columns[0]: 'Date', ecb_10.columns[1]: '10Y'}, inplace=True)
    ecb = pd.merge_ordered(ecb_1, ecb_10, on='Date')
    ecb.rename(columns={'1Y': '1Y_EU', '10Y': '10Y_EU'}, inplace=True)
    df = pd.merge_ordered(df_temp, ecb, on='Date')
    df.ffill(inplace=True)
    df = df.resample('D', on='Date').mean()
    df.ffill(inplace=True)
    return df


def preprocess_crypto_index_data(path: str,
                                 from_date: str = '2019-01-04',
                                 to_date: str = '2024-07-31',
                                 tickers: list = []
                                 ) -> pd.DataFrame:
    """
    This function preprocesses the crypto index data
    :param path: the path to the data
    :param from_date: date to start receiving the data (format: 'YYYY-MM-DD')
    :param to_date: date to end receiving the data (format: 'YYYY-MM-DD')
    :param tickers: the list of tickers to include in the data
    :return: the preprocessed data containing the crypto or index data (Close price)
    """
    df1 = pd.DataFrame()
    files = os.listdir(path)
    filtered_files = [file for file in files if any(ticker in file for ticker in tickers)]
    for file in filtered_files:
        df_temp = load_data(path + '/' + file, from_date=from_date, to_date=to_date)
        df_temp = df_temp.resample('D', on='Date').mean()
        df_temp.ffill(inplace=True)
        df_temp['Date'] = df_temp.index
        df_temp.reset_index(drop=True, inplace=True)
        df_temp = df_temp[['Date', 'Close']]
        df_temp.rename(columns={'Close': file.split('_')[-1].split('.')[0]}, inplace=True)
        if df1.empty:
            df1 = df_temp
            continue
        df1 = pd.merge_ordered(df1, df_temp, on='Date')
    return df1


def log_return(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    This function computes the log return of a column
    :param data: the dataframe with the column to compute the log return
    :param col_name: the column to compute the log return
    :return: the dataframe with the log return column
    """
    data['log_return_' + col_name] = np.log(data[col_name] / data[col_name].shift(1))
    return data


def dataframe_for_process(path: str, from_date: str = '2019-01-03',
                          to_date: str = '2024-07-31', save: bool = True) -> pd.DataFrame:
    """
    This function preprocesses the data and saves it to a csv file if save is True
    :param path: (str) The path to the data directory
    :param from_date: (str) The start date for the data (format: 'YYYY-MM-DD')
    :param to_date: (str) The end date for the data (format: 'YYYY-MM-DD')
    :param save: (bool) If True, the data is saved to a csv file
    :return: df (pd.DataFrame) The preprocessed data and saved to a csv file
    """
    df_temp = preprocess_yield_data(path=path)
    df_index = preprocess_crypto_index_data(path=path, from_date=from_date, to_date=to_date,
                                            characteristic='index_values')
    df_crypto = preprocess_crypto_index_data(path=path, from_date=from_date, to_date=to_date,
                                             characteristic='crypto_values')
    df_temp = pd.merge_ordered(df_temp, df_index, on='Date')
    df_temp = pd.merge_ordered(df_temp, df_crypto, on='Date')
    df3 = df_temp.copy()
    for col in df3.columns[1:9]:
        df3[f'first_diff_{col}'] = (df3[col].diff() / 100).round(7)
    for col in df3.columns[9:]:
        if 'first_diff' in col:
            continue
        df3 = log_return(df3, col)
    if save:
        from_date = pd.to_datetime(from_date) + pd.Timedelta(days=2)
        from_date = from_date.strftime('%Y-%m-%d')
        df3[df3['Date'] >= from_date].to_csv(
            f'{path}/processed_data_from_{from_date}_to_{to_date}.csv', index=False)
    return df3


def crypto_series_to_process(path: str, from_date: str = '2019-01-03',
                              to_date: str = '2024-07-31', save: bool = True, tickers: list = None) -> pd.DataFrame:
    """
    This function preprocesses the data and saves it to a csv file if save is True
    :param path: (str) The path to the data directory
    :param from_date: (str) The start date for the data (format: 'YYYY-MM-DD')
    :param to_date: (str) The end date for the data (format: 'YYYY-MM-DD')
    :param save: (bool) If True, the data is saved to a csv file
    :param tickers: (list) The list of tickers to include in the data
    :return: df (pd.DataFrame) The preprocessed data and saved to a csv file
    """
    if tickers is None:
        raise ValueError('Please provide a list of tickers to include in the data')

    df = preprocess_crypto_index_data(path=path, from_date=from_date, to_date=to_date, tickers=tickers)

    for col in df.columns:
        if "Date" in col:
            continue
        df = log_return(df, col)
    if save:
        from_date = pd.to_datetime(from_date) + pd.Timedelta(days=2)
        from_date = from_date.strftime('%Y-%m-%d')
        df[df['Date'] >= from_date].to_csv(
            f'{path}/processed_crypto_data_from_{from_date}_to_{to_date}.csv', index=False)
    return df

if __name__ == '__main__':
    base_path = os.getenv('BASE_PATH')
    if base_path is None:
        base_path = create_dir()
    start_date = '2019-01-03'
    end_date = '2024-07-31'
    save_file = True
    df = crypto_series_to_process(path=base_path, from_date=start_date, to_date=end_date, save=save_file)
    a = 3
