import numpy as np
from scipy.stats import norm


def compute_expected_sum_observed_R_if_observed_S_positive(
    N: int,
    corr_true: float,
    reward_scale: float,
    proxy_scale: float,
    corr_observed: float,
    units_per_arm: int,
    reward_noise_scale: float,
    proxy_noise_scale: float,
    folds: int,
) -> float:
    covar = corr_true * reward_scale * proxy_scale
    Lambda = np.array([[reward_scale**2, covar], [covar, proxy_scale**2]])

    noise_covar = reward_noise_scale * proxy_noise_scale * corr_observed
    Omega = np.array(
        [[reward_noise_scale**2, noise_covar], [noise_covar, proxy_noise_scale**2]]
    )

    total = Lambda + 2 / units_per_arm * Omega
    rho = total[0, 1] / np.sqrt(total[0, 0] * total[1, 1])

    return rho * np.sqrt(total[0, 0]) * norm.pdf(0) / (1 - norm.cdf(0)) * N * 0.5


def compute_expected_sum_true_R_if_observed_S_positive(
    N: int,
    corr_true: float,
    reward_scale: float,
    proxy_scale: float,
    corr_observed: float,
    units_per_arm: int,
    reward_noise_scale: float,
    proxy_noise_scale: float,
    folds: int,
) -> float:
    rho = (
        corr_true
        * proxy_scale
        / np.sqrt(proxy_scale**2 + 2 * proxy_noise_scale**2 / units_per_arm)
    )
    return rho * reward_scale * norm.pdf(0) / (1 - norm.cdf(0)) * N * 0.5


def compute_prob_obs_S_positive_if_true_R_positive(
    N: int,
    corr_true: float,
    reward_scale: float,
    proxy_scale: float,
    corr_observed: float,
    units_per_arm: int,
    reward_noise_scale: float,
    proxy_noise_scale: float,
    folds: int,
) -> float:
    rho = (
        corr_true
        * proxy_scale
        / np.sqrt(proxy_scale**2 + 2 * proxy_noise_scale**2 / units_per_arm)
    )
    return (1 / 4 + np.arcsin(rho) / np.pi / 2) / (1 / 2)


def compute_expected_sum_cross_fold_R_if_observed_S_positive(
    N: int,
    corr_true: float,
    reward_scale: float,
    proxy_scale: float,
    corr_observed: float,
    units_per_arm: int,
    reward_noise_scale: float,
    proxy_noise_scale: float,
    folds: int,
) -> float:
    if folds == 1:
        return compute_expected_sum_true_R_if_observed_S_positive(
            N,
            corr_true,
            reward_scale,
            proxy_scale,
            corr_observed,
            units_per_arm,
            reward_noise_scale,
            proxy_noise_scale,
        )
    scale = (folds - 1) / folds * units_per_arm
    rho = (
        corr_true
        * proxy_scale
        / np.sqrt(proxy_scale**2 + 2 * proxy_noise_scale**2 / scale)
    )
    return rho * reward_scale * norm.pdf(0) / (1 - norm.cdf(0)) * N * 0.5
