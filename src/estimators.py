from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.linalg import eig

from .plotting import inv_sqrtm


def get_beta_hat_from_vcov(vcov):
    """Fit OLS to a variance-covariance matrix.

    This is used to fit both the naive and total covariance (TC) estimators.
    """
    XtX = vcov[1:, 1:]
    XtY = vcov[0, 1:]
    return np.linalg.solve(XtX, XtY)


def get_last_eigenvector(matrix):
    """Get the last eigenvector (corresponding to the smallest eigenvalue) of a matrix."""
    eigenvalues, eigenvectors = eig(matrix)
    return eigenvectors[:, np.argsort(eigenvalues)[0]]


def get_beta_hat_from_tls(matrix, noise_vcov):
    """Fit Total Least Squares (TLS) to a rotated matrix."""
    matrix_rot = matrix @ inv_sqrtm(noise_vcov)
    last_eigenvector = get_last_eigenvector(matrix_rot.T @ matrix_rot)
    gamma = inv_sqrtm(noise_vcov) @ last_eigenvector
    return -gamma[1:] / gamma[0]


def get_beta_hat_from_vcov_minus_noise():
    """Given a set of"""


#### Code to implement our total covariance estimator
@dataclass
class ColumnMap:
    """Dataclass that maps column names to their role in our method(s).

    Usage:
    column_map = ColumnMap(
        test_id="test_id",
        outcome="y",
        mediators=["x1", "x2", "x3"],
        sample_size_treatment="n_t",
        sample_size_control="n_c",
    )
    """

    test_id: str
    outcome: str
    mediators: List[str]
    sample_size_treatment: str
    sample_size_control: str


def get_effective_sample_size(
    dataframe: pd.DataFrame, column_map: ColumnMap, as_np=True
):
    """Returns 1 / (1 / n_treatment + 1 / n_control)."""
    n_eff = 1 / (
        1 / dataframe[column_map.sample_size_treatment]
        + 1 / dataframe[column_map.sample_size_control]
    )
    if as_np:
        return n_eff.values
    return n_eff


def get_total_vcov(dataframe: pd.DataFrame, column_map: ColumnMap):
    """Estimate total covariance of estimated treatment effects on a set of metrics.

    The total covariance of estimated treatment effects is equal to the covariance in
    true treatment effects plus scaled correlated measurement error.  This implies a
    simple estimator for the covariance in true treatment effects that subtracts an
    estimate of the scaled correlated measurement error from the total covariance.
    """
    _dataframe = dataframe.copy()
    n_eff = get_effective_sample_size(_dataframe, column_map)
    metrics = [column_map.outcome, *column_map.mediators]

    # Step 1: Compute the weighted average by effective_sample_size for each metric
    weighted_averages = {}
    for m in metrics:
        weighted_avg = np.sum(_dataframe[m] * n_eff) / np.sum(n_eff)
        weighted_averages[m] = weighted_avg

    # Step 2: Subtract the weighted average from each treatment effect column
    for m in metrics:
        _dataframe[m] -= weighted_averages[m]

    # Step 3: Compute the weighted covariance matrix of the treatment effects
    weighted_cov_matrix = np.cov(
        _dataframe[metrics].values,
        aweights=n_eff,
        rowvar=False,
    )

    return weighted_cov_matrix


def sum_product_over_tests(dataframe: pd.DataFrame, column_map: ColumnMap):
    total_sum = 0

    # Iterate over unique test_ids
    for test_id in dataframe[column_map.test_id].unique():
        # Filter rows for the current test_id
        test_data = dataframe[dataframe[column_map.test_id] == test_id]

        # Extract count_control (assuming it's the same for all cells in the same test)
        count_control = test_data[
            column_map.sample_size_control or column_map.sample_size_treatment
        ].iloc[0]

        # Iterate over distinct pairs of cells within the same test
        for i in range(len(test_data)):
            for j in range(i + 1, len(test_data)):
                total_sum += (
                    test_data[column_map.sample_size_treatment].iloc[i]
                    * test_data[column_map.sample_size_treatment].iloc[j]
                ) / count_control

    return total_sum


def get_total_vcov_minus_noise(
    dataframe: pd.DataFrame,
    noise_matrix: np.array,
    column_map: ColumnMap,
    scale: float = 1.0,
):
    n_eff = get_effective_sample_size(dataframe, column_map)
    total_vcov = get_total_vcov(dataframe, column_map)
    products = sum_product_over_tests(dataframe, column_map)
    return (
        total_vcov
        - noise_matrix
        * scale
        * (len(dataframe) - 1 / sum(n_eff) - products / sum(n_eff))
        / sum(n_eff)
    ) / (1 - sum(n_eff**2) / (sum(n_eff) ** 2))


def get_beta_hat_from_vcov_minus_noise(
    dataframe,
    noise_matrix,
    column_map,
    scale: float = 1.0,
):
    return get_beta_hat_from_vcov(
        get_total_vcov_minus_noise(dataframe, noise_matrix, column_map, scale)
    )
