import pandas as pd
import numpy as np
import os
from cvxopt import matrix
from cvxopt.solvers import qp
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def style_analysis_quadratic_programming(data: pd.DataFrame, index_columns: list, target_variable: str) -> np.array:
    """
    This function solves a quadratic programming problem

    target function: min -x^T Q x

    x = [x0, x1, x2, ..., xn] where x0 + x1 + x2 + ... + xn-1 = 1 and xn =-1 and xi >= 0 for i = 0, 1, 2, ..., n-1

    Q is the covariance matrix of the columns in the indexes list + target variable (as the last column)

    ========
    Example:
    ========
        general form:
                    L(x) = -x^T r + x^T Q x \n
                    with constraints: \n
                    Gx <= h \n
                    Ax = b \n

        # Generate random vector r and symmetric definite positive matrix Q \n
        n = 50 \n
        r = matrix(np.random.sample(n)) \n
        Q = np.random.randn(n,n) \n
        Q = 0.5 * (Q + Q.T) \n
        Q = Q + n * np.eye(n) \n
        Q = matrix(Q) \n
        # Add constraint matrices and vectors \n
        A = matrix(np.ones(n)).T \n
        b = matrix(1.0) \n
        G = matrix(- np.eye(n)) \n
        h = matrix(np.zeros(n)) \n

        # Solve and retrieve solution

        sol = qp(Q, -r, G, h, A, b)['x']
    ========================================================
    :param data: (pd.DataFrame) the dataframe with the data to solve the problem
    :param index_columns: (list) ist of columns containing the index values for the sharpe style analysis
    :param target_variable: (str) the target variable for the sharpe style analysis (the variable to perform the analysis)
    :return: (np.array) the coefficients of the solution
    """

    q_columns = index_columns + [target_variable]
    covariance_matrix = data[q_columns].cov().values
    n = len(covariance_matrix)
    r = matrix(np.zeros(n))
    Q = matrix(covariance_matrix)
    # constraints
    a = np.ones((n, 2))
    a[:-1, 1] = 0
    a[-1, 0] = 0
    A = matrix(a).T
    b = matrix([1.0, -1.0])
    G = matrix(- np.eye(n))
    h = matrix(np.append(np.zeros(n - 1), [1], axis=0))
    # Solve and retrieve solution
    sol = qp(-Q, r, G, h, A, b)['x']
    return np.array(sol)


def style_analysis_confidence_intervals(data: pd.DataFrame, index_columns: list,
                                        target_variable: str, coeffs: np.array) -> np.array:
    """
    Apply the quadratic programming function to the dataframe using a rolling window approach.

    :param data: (pd.DataFrame) The dataframe with the data to solve the problem.
    :param index_columns: (list) List of columns containing the index values for the Sharpe style analysis.
    :param target_variable: (str) The target variable for the Sharpe style analysis (the variable to perform the analysis).
    :param coeffs: (np.array) the coefficients of the solution
    :return: (pd.DataFrame) DataFrame containing the confidence interval for each weight assigned to each index .
    """
    coeffs[coeffs < 1e-2] = 0
    k = np.count_nonzero(coeffs)
    n = len(data)
    error = data[target_variable].to_numpy() - np.squeeze(np.dot(data[index_columns].to_numpy(), coeffs))
    sigma_alpha = np.var(error) * (n - k) / (n-1)
    confidence_interval_list = []
    for idx, column in enumerate(index_columns):
        if coeffs[idx] == 0:
            confidence_interval_list.append(0)
            continue
        columns_list = index_columns.copy()
        columns_list.remove(column)
        solution = style_analysis_quadratic_programming(data, columns_list, target_variable=column)
        b_i= data[column].to_numpy() - np.squeeze(np.dot(data[columns_list].to_numpy(), solution[:-1]))
        sigma_b_i = np.var(b_i)
        confidence_interval = sigma_alpha /(sigma_b_i * np.sqrt(n - k - 1))
        confidence_interval_list.append(confidence_interval)

    return np.array(confidence_interval_list)


def compute_r_squared(data: pd.DataFrame, index_columns: list, target_variable: str, coeffs: np.array) -> float:
    """
    This function computes the R-squared of the Sharpe Style Analysis

    Formula:
    R^2 = 1 - Var(e) / Var(R) where e: error term and R: returns of target variable

    :param data: (pd.DataFrame) the dataframe with the data to solve the problem
    :param index_columns: (list) ist of columns containing the index values for the sharpe style analysis
    :param target_variable: (str) the target variable for the sharpe style analysis (the variable to perform the analysis)
    :param coeffs: (np.array) the coefficients of the solution
    :return: (float) the R-squared of the Sharpe Style Analysis
    """
    error = data[target_variable].to_numpy() - np.squeeze(np.dot(data[index_columns].to_numpy(), coeffs))
    r_squared = 1 - np.var(error) / np.var(data[target_variable].to_numpy())
    return r_squared


def rolling_style_analysis(data: pd.DataFrame, index_columns: list, target_variable: str,
                           window: int, step: int) -> pd.DataFrame:
    """
    Apply the quadratic programming function to the dataframe using a rolling window approach.

    :param data: (pd.DataFrame) The dataframe with the data to solve the problem.
    :param index_columns: (list) List of columns containing the index values for the Sharpe style analysis.
    :param target_variable: (str) The target variable for the Sharpe style analysis (the variable to perform the analysis).
    :param window: (int) The size of the rolling window (in days).
    :param step: (int) The step size for the rolling window (in days).
    :return: (pd.DataFrame) DataFrame containing the coefficients for each rolling window.
    """
    solutions = []
    ci_list = []
    r_value = []
    dates_idx = []

    for start in range(0, len(data) - window + 1, step):
        end = start + window
        if end >= len(data):
            break
        dates_idx.append(end)
        window_data = data.iloc[start:end]
        solution = style_analysis_quadratic_programming(window_data, index_columns, target_variable)
        conf_interval = style_analysis_confidence_intervals(window_data, index_columns, target_variable, solution[:-1])
        solutions.append(np.squeeze(solution[:-1]))
        ci_list.append(np.squeeze(conf_interval))
        r_value.append(compute_r_squared(window_data, index_columns, target_variable, solution[:-1, 0]))

    r_value_df = pd.DataFrame(r_value, columns=['R_squared'])
    solutions_df = pd.DataFrame(solutions, columns=[f'coeff_{i}' for i in range(len(solutions[0]))])
    ci_df = pd.DataFrame(ci_list, columns=[f'conf_interval_{i}' for i in range(len(ci_list[0]))])
    solutions_df = pd.concat([solutions_df, ci_df, r_value_df], axis=1)
    solutions_df.index = data['Date'].iloc[dates_idx]
    return solutions_df


if __name__ == '__main__':
    base_path = os.getenv('BASE_PATH')
    df = pd.read_csv(f'{base_path}/processed_data_from_2019-01-05_to_2024-07-31.csv', parse_dates=['Date'])
    cols = ['log_return_BTC-USD', 'log_return_ETH-USD', 'log_return_GC=F'] # [col for col in df.columns if 'first_diff' in col or ('^' in col and 'log_return' in col)] +
    target = 'log_return_BNB-USD'
    sol = rolling_style_analysis(data=df, index_columns=cols, target_variable=target, window=120, step=1)
    a = 3