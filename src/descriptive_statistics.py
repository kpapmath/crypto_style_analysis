import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv, find_dotenv

from src.preprocess_data import create_dir

load_dotenv(find_dotenv())


def series_describe(data: pd.DataFrame, column_name: str) -> pd.Series:
    """
    Compute an extensive list of  descriptive statistics for time series

    :param data: (pd.DataFrame) DataFrame containing the time series data.
    :param column_name: (str) Name of the column to compute the descriptive statistics.
    :return: (pd.Series) Series containing the descriptive statistics of the time series data for the specified column.
    """
    res = pd.Series(index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis', 'autocorr',
                           'drawdown', 'drawup','drawdown_percent', 'drawup_percent', ])
    # to avoid error when the column is datetime type we create a series of only datetime type
    res_date = pd.Series(index=[ 'drawdown_start','drawup_start', 'drawdown_end', 'drawup_end'], dtype='datetime64[ns]')

    series = data[column_name]
    res['count'] = series.count()
    res['mean'] = series.mean()
    res['std'] = series.std()
    res['min'] = series.min()
    res['25%'] = series.quantile(0.25)
    res['50%'] = series.quantile(0.5)
    res['75%'] = series.quantile(0.75)
    res['max'] = series.max()
    res['skew'] = series.skew()
    res['kurtosis'] = series.kurtosis()
    res['autocorr'] = series.autocorr()
    if 'log_return' in column_name or 'Y' in column_name:
        drawdown, drawdown_start, drawdown_end = np.nan, 0, 0
        drawup, drawup_start, drawup_end = np.nan, 0, 0
        res_date['drawdown_start'] = np.datetime64("NaT")
        res_date['drawup_start'] = np.datetime64("NaT")
        res_date['drawdown_end'] = np.datetime64("NaT")
        res_date['drawup_end'] = np.datetime64("NaT")
        res['drawdown_percent'] = np.nan
        res['drawup_percent'] = np.nan
    else:
        drawdown, drawdown_start, drawdown_end = compute_draw_down_up(series, draw_down=True)
        drawup, drawup_start, drawup_end = compute_draw_down_up(series, draw_down=False)
        res_date['drawdown_start'] = data['Date'].iloc[drawdown_start]
        res_date['drawup_start'] = data['Date'].iloc[drawup_start]
        res_date['drawdown_end'] = data['Date'].iloc[drawdown_end]
        res_date['drawup_end'] = data['Date'].iloc[drawup_end]
        res['drawdown_percent'] = drawdown / series[drawdown_start]
        res['drawup_percent'] = drawup / series[drawup_start]
    res['drawdown'] = drawdown
    res['drawup'] = drawup
    res['drawdown_duration'] = data['Date'].iloc[drawdown_end] - data['Date'].iloc[drawdown_start]
    res['drawup_duration'] = data['Date'].iloc[drawup_end] - data['Date'].iloc[drawup_start]

    res = pd.concat([res, res_date])
    return res[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis', 'autocorr', 'drawdown',
                'drawdown_percent', 'drawdown_duration', 'drawdown_start', 'drawdown_end', 'drawup',
                'drawup_percent', 'drawup_duration', 'drawup_start', 'drawup_end']]


def compute_draw_down_up(series, draw_down=True) -> tuple:
    """
    Compute the draw-down and draw-up of a time series.

    :param series: (pd.Series) Series containing the time series data.
    :param draw_down: (bool) If True, compute the draw-down; otherwise, compute the draw-up.
    :return: (tuple) The maximum draw-down (maximum draw-up), start time index, and end time index.
    """
    max_value = series[0]
    max_drawdown = 0
    min_value = series[0]
    max_drawup = 0
    index_start = series.index[0]
    index_end = series.index[0]
    temp_start = series.index[0]
    if draw_down:
        for i in range(1, len(series)):
            if series[i] > max_value:
                max_value = series[i]
                temp_start = series.index[i]
            drawdown = max_value - series[i]
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                index_start = temp_start
                index_end = series.index[i]
        value = max_drawdown
    else:
        for i in range(1, len(series)):
            if series[i] < min_value:
                min_value = series[i]
                temp_start = series.index[i]
            drawup = series[i] - min_value
            if drawup > max_drawup:
                max_drawup = drawup
                index_start = temp_start
                index_end = series.index[i]
        value = max_drawup

    return value, index_start, index_end


def descriptive_statistics_function(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the descriptive statistics of the time series data.

    :param data: (pd.DataFrame) DataFrame containing the time series data.
    :return: (pd.DataFrame) DataFrame containing the descriptive statistics of the time series data.
    """
    res = pd.DataFrame()
    for column in data.columns:
        if column == 'Date':
            continue
        res[column] = series_describe(data, column)
    return res


if __name__ == '__main__':
    base_path = os.getenv('BASE_PATH')
    if base_path is None:
        base_path = create_dir()
    df = pd.read_csv(f'{base_path}/processed_data.csv', parse_dates=['Date'])
    result = descriptive_statistics_function(df)
    a=3
