# Functions for simulating data
import numpy as np
import pandas as pd

from .estimators import get_beta_hat_from_vcov, get_beta_hat_from_tls


def simulate_metric_means(sample_size, rng, unit_covariance, means=None):
    return rng.multivariate_normal(
        mean=means if means is not None else np.array([0.0, 0.0, 0]),
        cov=unit_covariance / sample_size,
    )


def simulate_treatment_effects(
    sample_size,
    rng,
    unit_covariance,
    treatment_covariance,
    surrogacy_params,
    direct_effects=False,
):
    treatment_means = rng.multivariate_normal(
        mean=np.array([0.0, 0.0, 0.0]),
        cov=treatment_covariance,
    )
    if not direct_effects:
        treatment_means[0] = treatment_means[1:] @ surrogacy_params
    treatment = simulate_metric_means(
        sample_size, rng, unit_covariance, treatment_means
    )
    control = simulate_metric_means(sample_size, rng, unit_covariance)
    return treatment - control


def simulate_many_treatment_effects(
    sample_size,
    rng,
    unit_covariance,
    treatment_covariance,
    surrogacy_params,
    direct_effects=False,
    times=1_000,
):
    return pd.DataFrame(
        np.array(
            [
                simulate_treatment_effects(
                    sample_size,
                    rng,
                    unit_covariance,
                    treatment_covariance,
                    surrogacy_params,
                    direct_effects,
                )
                for _ in range(times)
            ]
        ),
        columns=["y", "x1", "x2"],
    ).assign(sample_size=sample_size)


def get_estimates_for_sample_size(sample_size, data, unit_covariance, surrogacy_params):
    subset = data[data.sample_size == sample_size][["y", "x1", "x2"]].values
    liml = get_beta_hat_from_tls(subset, unit_covariance)
    tc = get_beta_hat_from_vcov(
        np.cov(subset, rowvar=False) - 2 * unit_covariance / sample_size
    )
    naive = get_beta_hat_from_vcov(np.cov(subset, rowvar=False))
    return pd.DataFrame(
        {
            "sample_size": [sample_size] * 4,
            "estimator": ["truth", "liml", "tc", "naive"],
            "beta_1": [surrogacy_params[0], liml[0], tc[0], naive[0]],
            "beta_2": [surrogacy_params[1], liml[1], tc[1], naive[1]],
        }
    )


def get_estimates(
    draw,
    sample_sizes,
    rng,
    unit_covariance,
    treatment_covariance,
    surrogacy_params,
    direct_effects=False,
):
    data = pd.concat(
        [
            simulate_many_treatment_effects(
                n,
                rng,
                unit_covariance,
                treatment_covariance,
                surrogacy_params,
                direct_effects,
                times=1_000,
            )
            for n in sample_sizes
        ]
    )
    return pd.concat(
        [
            get_estimates_for_sample_size(
                s,
                data,
                unit_covariance,
                surrogacy_params,
            )
            for s in sample_sizes
        ]
    ).assign(draw=draw)
