import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from tqdm import tqdm

from constants import (
    NUM_SIMULATIONS,
    DEFAULT_FOLDS,
    DEFAULT_N,
    DEFAULT_REWARD_SCALE,
    DEFAULT_PROXY_SCALE,
    DEFAULT_UNITS_PER_ARM,
    DEFAULT_REWARD_NOISE_SCALE,
    DEFAULT_CORR_TRUE,
    DEFAULT_CORR_OBSERVED,
    SEED,
)
from functions import (
    compute_expected_sum_true_R_if_observed_S_positive,
    compute_expected_sum_observed_R_if_observed_S_positive,
    compute_expected_sum_cross_fold_R_if_observed_S_positive,
)
from simulation import (
    simulate_true_treatment_effects,
    simulate_observed_treatment_effects,
    estimate_treatment_sums,
)


proxy_scale_values = [DEFAULT_PROXY_SCALE]
proxy_noise_values = DEFAULT_PROXY_SCALE * np.array([10, 50, 100, 250, 500, 1000])


results = {
    "sim_naive_bias": {},
    "sim_cv_bias": {},
    "theo_naive_bias": {},
    "theo_cv_bias": {},
}
x_axis = np.array(proxy_noise_values)

for pscale in proxy_scale_values:

    sim_naive_list = []
    sim_cv_list = []
    theo_naive_list = []
    theo_cv_list = []

    for pnoise in proxy_noise_values:

        target = compute_expected_sum_true_R_if_observed_S_positive(
            N=DEFAULT_N,
            corr_true=DEFAULT_CORR_TRUE,
            reward_scale=DEFAULT_REWARD_SCALE,
            proxy_scale=pscale,
            corr_observed=DEFAULT_CORR_OBSERVED,
            units_per_arm=DEFAULT_UNITS_PER_ARM,
            reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
            proxy_noise_scale=pnoise,
            folds=DEFAULT_FOLDS,
        )

        theory_naive = compute_expected_sum_observed_R_if_observed_S_positive(
            N=DEFAULT_N,
            corr_true=DEFAULT_CORR_TRUE,
            reward_scale=DEFAULT_REWARD_SCALE,
            proxy_scale=pscale,
            corr_observed=DEFAULT_CORR_OBSERVED,
            units_per_arm=DEFAULT_UNITS_PER_ARM,
            reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
            proxy_noise_scale=pnoise,
            folds=DEFAULT_FOLDS,
        )
        theo_naive_bias = theory_naive / target - 1

        theory_cv = compute_expected_sum_cross_fold_R_if_observed_S_positive(
            N=DEFAULT_N,
            corr_true=DEFAULT_CORR_TRUE,
            reward_scale=DEFAULT_REWARD_SCALE,
            proxy_scale=pscale,
            corr_observed=DEFAULT_CORR_OBSERVED,
            units_per_arm=DEFAULT_UNITS_PER_ARM,
            reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
            proxy_noise_scale=pnoise,
            folds=DEFAULT_FOLDS,
        )
        theo_cv_bias = theory_cv / target - 1

        sim_naive_biases = []
        sim_cv_biases = []

        for sim in tqdm(range(NUM_SIMULATIONS)):

            true_treatment_effects = simulate_true_treatment_effects(
                number_of_experiments=DEFAULT_N,
                corr=DEFAULT_CORR_TRUE,
                reward_scale=DEFAULT_REWARD_SCALE,
                proxy_scale=pscale,
                seed=SEED + sim,
            )

            observed_treatment_effects = simulate_observed_treatment_effects(
                true_treatment_effects=true_treatment_effects,
                units_per_arm=DEFAULT_UNITS_PER_ARM,
                reward_noise_scale=DEFAULT_REWARD_NOISE_SCALE,
                proxy_noise_scale=pnoise,
                corr=DEFAULT_CORR_OBSERVED,
                folds=DEFAULT_FOLDS,
            )

            sums = estimate_treatment_sums(
                true_treatment_effects, observed_treatment_effects
            )

            naive_sum = sums["sum_observed_R_if_observed_S_positive"]
            cv_sum = sums["sum_cross_fold_R_if_observed_S_positive"]

            sim_naive_biases.append(naive_sum / target - 1)
            sim_cv_biases.append(cv_sum / target - 1)

        sim_naive_mean_bias = np.mean(sim_naive_biases)
        sim_cv_mean_bias = np.mean(sim_cv_biases)

        sim_naive_list.append(sim_naive_mean_bias)
        sim_cv_list.append(sim_cv_mean_bias)
        theo_naive_list.append(theo_naive_bias)
        theo_cv_list.append(theo_cv_bias)

    results["sim_naive_bias"][pscale] = np.array(sim_naive_list)
    results["sim_cv_bias"][pscale] = np.array(sim_cv_list)
    results["theo_naive_bias"][pscale] = np.array(theo_naive_list)
    results["theo_cv_bias"][pscale] = np.array(theo_cv_list)

x_axis_scaled = x_axis / DEFAULT_PROXY_SCALE


plt.figure(figsize=(5, 3))
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
for pscale in proxy_scale_values:

    plt.plot(
        x_axis_scaled,
        results["sim_naive_bias"][pscale],
        marker="x",
        linestyle="--",
        label="Naive Estimator",
        color="red",
    )
    plt.plot(
        x_axis_scaled,
        results["sim_cv_bias"][pscale],
        marker="o",
        label="Cross-Validation Estimator",
        color="blue",
    )
plt.title("Relative Bias versus Proxy Noise-to-Signal Ratio")
plt.xlabel("Proxy Noise-to-Signal Ratio")
plt.ylabel("Relative Bias of Estimated Reward")


plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))


plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}x"))

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("figs/figure-2.png")
