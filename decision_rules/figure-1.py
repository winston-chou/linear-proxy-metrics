from matplotlib import pyplot as plt
import numpy as np

from constants import (
    DEFAULT_FOLDS,
    DEFAULT_N,
    DEFAULT_REWARD_SCALE,
    DEFAULT_PROXY_SCALE,
    DEFAULT_UNITS_PER_ARM,
    DEFAULT_REWARD_NOISE_SCALE,
    DEFAULT_PROXY_NOISE_SCALE,
    GOOD_PROXY,
    BAD_PROXY,
)
from functions import (
    compute_expected_sum_true_R_if_observed_S_positive,
    compute_expected_sum_observed_R_if_observed_S_positive,
    compute_expected_sum_cross_fold_R_if_observed_S_positive,
)


def plot_heatmap(to_plot, title, filename, vmax=None):
    plt.figure(figsize=(5, 4))
    if not vmax:
        vmax = np.abs(to_plot).max()
    plt.imshow(
        to_plot,
        extent=[
            corr_observeds.min(),
            corr_observeds.max(),
            corr_trues.min(),
            corr_trues.max(),
        ],
        origin="lower",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    cbar = plt.colorbar(label="Expected Value")
    cbar.set_ticks([])

    plt.title(title)
    plt.xlabel("Correlation in Measurement Error")
    plt.ylabel("Correlation in Treatment Effects")

    plt.text(*GOOD_PROXY[::-1], "G", size=14, weight="bold")
    plt.text(*BAD_PROXY[::-1], "B", size=14, weight="bold")

    plt.tight_layout()
    plt.savefig(filename)


n_points = 20
corr_trues = np.linspace(-1.0, 1.0, n_points)
corr_observeds = np.linspace(-1.0, 1.0, n_points)


to_plot_true = np.zeros((n_points, n_points))
to_plot_observed = np.zeros((n_points, n_points))
to_plot_cross_fold = np.zeros((n_points, n_points))


for i, ct in enumerate(corr_trues):
    for j, co in enumerate(corr_observeds):
        to_plot_true[i, j] = compute_expected_sum_true_R_if_observed_S_positive(
            N=DEFAULT_N,
            corr_true=ct,
            reward_scale=DEFAULT_REWARD_SCALE,
            proxy_scale=DEFAULT_PROXY_SCALE,
            corr_observed=co,
            units_per_arm=DEFAULT_UNITS_PER_ARM,
            reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
            proxy_noise_scale=DEFAULT_PROXY_NOISE_SCALE,
            folds=DEFAULT_FOLDS,
        )
        to_plot_observed[i, j] = compute_expected_sum_observed_R_if_observed_S_positive(
            N=DEFAULT_N,
            corr_true=ct,
            reward_scale=DEFAULT_REWARD_SCALE,
            proxy_scale=DEFAULT_PROXY_SCALE,
            corr_observed=co,
            units_per_arm=DEFAULT_UNITS_PER_ARM,
            reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
            proxy_noise_scale=DEFAULT_PROXY_NOISE_SCALE,
            folds=DEFAULT_FOLDS,
        )
        to_plot_cross_fold[
            i, j
        ] = compute_expected_sum_cross_fold_R_if_observed_S_positive(
            N=DEFAULT_N,
            corr_true=ct,
            reward_scale=DEFAULT_REWARD_SCALE,
            proxy_scale=DEFAULT_PROXY_SCALE,
            corr_observed=co,
            units_per_arm=DEFAULT_UNITS_PER_ARM,
            reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
            proxy_noise_scale=DEFAULT_PROXY_NOISE_SCALE,
            folds=2,
        )


plot_heatmap(to_plot_true, "Expectation of True Reward", "figs/figure-1-1.png")
plot_heatmap(
    to_plot_observed,
    "Expectation of Naive Estimator",
    "figs/figure-1-2.png",
)
plot_heatmap(
    to_plot_cross_fold,
    "Expectation of Cross-Validation Estimator",
    "figs/figure-1-3.png",
    vmax=np.abs(to_plot_true).max(),
)
