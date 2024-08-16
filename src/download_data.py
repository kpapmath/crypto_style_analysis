import pandas as pd
import yfinance as yf
import os
import fredapi as fred
from dotenv import load_dotenv, find_dotenv

from src.preprocess_data import create_dir

load_dotenv(find_dotenv())


def download_data_yahoo(path: str, ticker_name: str, from_date: str, to_date: str,
                        save_csv: bool = True) -> pd.DataFrame:
    """
    Download data from Yahoo Finance API

    :param path: the path to save the data
    :param ticker_name: the ticker name of the series
    :param from_date:  date to start receiving the data (format: 'YYYY-MM-DD')
    :param to_date:  date to end receiving the data (format: 'YYYY-MM-DD')
    :param save_csv: save the data to a csv file
    :return:  the data in a pandas DataFrame
    """
    data = yf.download(ticker_name, start=from_date, end=to_date)
    print(data.head())
    if data.empty:
        print(f'No data found for series {ticker_name}')
        raise ValueError
    if save_csv:
        if '^' in ticker_name or '000' in ticker_name:
            prefix = 'index_values'
        else:
            prefix = 'crypto_values'
        data.to_csv(f'{path}/{prefix}_from_{from_date}_to_{to_date}_{ticker_name}.csv')

    return data


def download_data_fred(path: str, fred_ticker: str, from_date: str, to_date: str,
                       save_csv: bool = True) -> pd.DataFrame:
    """
    Download data from FRED API

    :param path: the path to save the data
    :param fred_ticker: the series id from FRED
    :param from_date:  date to start receiving the data (format: 'YYYY-MM-DD')
    :param to_date: date to end receiving the data (format: 'YYYY-MM-DD')
    :param save_csv:  save the data to a csv file
    :return:  the data in a pandas DataFrame
    """
    fred_api = fred.Fred(api_key=os.getenv('FRED_API_KEY'))
    data = fred_api.get_series(fred_ticker, observation_start=from_date, observation_end=to_date)
    print(data.head())
    if data.empty:
        print(f'No data found for series {fred_ticker}')
        raise ValueError
    data = pd.DataFrame(data, columns=[fred_ticker])
    data['Date'] = data.index
    if save_csv:
        data[['Date', fred_ticker]].to_csv(f'{path}/fred_data_from_{from_date}_to_{to_date}_{fred_ticker}.csv',
                                           index=False)
    return data


def download_data(path: str, tickers: list = None, fred_tickers: list = None, from_date: str = None, to_date: str = None,
                  save_csv: bool = True):
    """
    Download data from Yahoo Finance and FRED API and save the files to CSV 's

    :param path: the path to save the data
    :param tickers: a list of ticker names
    :param fred_tickers: a list of series ids from FRED
    :param from_date: date to start receiving the data (format: 'YYYY-MM-DD')
    :param to_date: date to end receiving the data (format: 'YYYY-MM-DD')
    :param save_csv: save the data to a csv file
    :return: None
    """
    if tickers is None and fred_tickers is None:
        raise ValueError('No tickers and series are provided')

    if len(tickers) == 0:
        print('No tickers provided')
    else:
        for name in tickers:
            print(f'Downloading data for the following tickers: {name}')
            download_data_yahoo(path, name, from_date, to_date, save_csv)
    if len(fred_tickers) == 0:
        print('No fred tickers provided')
    else:
        for series_id in fred_tickers:
            print(f'Downloading data for the following series: {series_id}')
            download_data_fred(path, series_id, from_date, to_date, save_csv)


if __name__ == '__main__':
    base_dir = os.getenv('BASE_PATH')
    if base_dir is None:
        base_dir = create_dir()
    start_date = '2019-01-01'
    end_date = '2024-08-12'
    ticker = ['^HSI', '^DJI', '^IXIC', '^N225', '^STOXX', 'ADA-USD', 'BTC-USD', 'ETH-USD',
              'XRP-USD', 'LTC-USD', 'BNB-USD', 'DOGE-USD', 'GC=F']
    series = ['DGS1', 'DGS10']
    save_file = True
    download_data(path=base_dir, tickers=ticker, fred_tickers=series, from_date=start_date, to_date=end_date,
                  save_csv=save_file)
    a = 3
