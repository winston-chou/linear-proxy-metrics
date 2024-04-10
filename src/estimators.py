import numpy as np
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
