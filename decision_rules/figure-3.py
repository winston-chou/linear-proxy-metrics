import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from constants import (
    DEFAULT_N,
    DEFAULT_REWARD_SCALE,
    DEFAULT_PROXY_SCALE,
    DEFAULT_REWARD_NOISE_SCALE,
    DEFAULT_PROXY_NOISE_SCALE,
    GOOD_PROXY,
    BAD_PROXY,
    DEFAULT_FOLDS,
    NUM_SIMULATIONS,
)
from functions import (
    compute_expected_sum_true_R_if_observed_S_positive,
)
from simulation import (
    simulate_true_treatment_effects,
    simulate_observed_treatment_effects,
    estimate_treatment_sums,
)


params_A = {
    "corr_true": GOOD_PROXY[0],
    "reward_scale": DEFAULT_REWARD_SCALE,
    "proxy_scale": DEFAULT_PROXY_SCALE,
    "corr_observed": GOOD_PROXY[1],
    "reward_noise_scale": DEFAULT_REWARD_NOISE_SCALE,
    "proxy_noise_scale": DEFAULT_PROXY_NOISE_SCALE,
    "folds": DEFAULT_FOLDS,
}


params_B = {
    "corr_true": BAD_PROXY[0],
    "reward_scale": DEFAULT_REWARD_SCALE,
    "proxy_scale": DEFAULT_PROXY_SCALE,
    "corr_observed": BAD_PROXY[1],
    "reward_noise_scale": DEFAULT_REWARD_NOISE_SCALE,
    "proxy_noise_scale": DEFAULT_PROXY_NOISE_SCALE,
    "folds": DEFAULT_FOLDS,
}


units_per_arm_values = [1_000_000, 2_000_000, 5_000_000, 10_000_000]


def run_scenario(param_dict, units_per_arm, N, num_sims):
    """
    For a given scenario's param_dict and a given units_per_arm,
    run 'num_sims' random simulations and return the average naive
    and CV sums of R if S>0.
    """

    naive_sums = []
    cv_sums = []

    target = compute_expected_sum_true_R_if_observed_S_positive(
        N=DEFAULT_N,
        corr_true=param_dict["corr_true"],
        reward_scale=param_dict["reward_scale"],
        proxy_scale=param_dict["proxy_scale"],
        corr_observed=param_dict["corr_observed"],
        units_per_arm=units_per_arm,
        reward_noise_scale=param_dict["reward_noise_scale"],
        proxy_noise_scale=param_dict["proxy_noise_scale"],
        folds=param_dict["folds"],
    )

    for sim in tqdm(range(num_sims)):

        true_effects = simulate_true_treatment_effects(
            number_of_experiments=DEFAULT_N,
            corr=param_dict["corr_true"],
            reward_scale=param_dict["reward_scale"],
            proxy_scale=param_dict["proxy_scale"],
            seed=1000 + sim,
        )

        observed_effects = simulate_observed_treatment_effects(
            true_treatment_effects=true_effects,
            units_per_arm=units_per_arm,
            reward_noise_scale=param_dict["reward_noise_scale"],
            proxy_noise_scale=param_dict["proxy_noise_scale"],
            corr=param_dict["corr_observed"],
            folds=param_dict["folds"],
        )

        sums = estimate_treatment_sums(true_effects, observed_effects)
        naive_sums.append(sums["sum_observed_R_if_observed_S_positive"])
        cv_sums.append(sums["sum_cross_fold_R_if_observed_S_positive"])

    return target, np.mean(naive_sums), np.mean(cv_sums)


target_A = []
naive_A = []
cv_A = []
target_B = []
naive_B = []
cv_B = []

for M in units_per_arm_values:

    a_target, a_naive, a_cv = run_scenario(params_A, M, DEFAULT_N, NUM_SIMULATIONS)
    target_A.append(a_target)
    naive_A.append(a_naive)
    cv_A.append(a_cv)

    b_target, b_naive, b_cv = run_scenario(params_B, M, DEFAULT_N, NUM_SIMULATIONS)
    target_B.append(b_target)
    naive_B.append(b_naive)
    cv_B.append(b_cv)


plt.figure(figsize=(5, 9))


plt.subplot(3, 1, 1)
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.plot(
    units_per_arm_values, target_A, marker="o", color="tab:blue", label="Good Proxy"
)
plt.plot(
    units_per_arm_values, target_B, marker="s", color="tab:orange", label="Bad Proxy"
)

plt.xlabel("Units per Treatment Arm")
plt.ylabel("True Cumulative Reward")
plt.title("True Cumulative Reward")
plt.grid(True)
plt.ylim([-0.001, 0.004])
plt.legend()

plt.subplot(3, 1, 2)
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.plot(
    units_per_arm_values,
    target_A,
    marker="o",
    color="tab:blue",
    linestyle="--",
    alpha=0.40,
)
plt.plot(
    units_per_arm_values, naive_A, marker="o", color="tab:blue", label="Good Proxy"
)
plt.plot(
    units_per_arm_values,
    target_B,
    marker="s",
    color="tab:orange",
    linestyle="--",
    alpha=0.40,
)
plt.plot(
    units_per_arm_values, naive_B, marker="s", color="tab:orange", label="Bad Proxy"
)

plt.xlabel("Units per Treatment Arm")
plt.ylabel("Estimated Cumulative Reward")
plt.title("Naive Estimator")
plt.grid(True)
plt.ylim([-0.001, 0.004])
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(
    units_per_arm_values,
    target_A,
    marker="o",
    color="tab:blue",
    linestyle="--",
    alpha=0.40,
)
plt.plot(units_per_arm_values, cv_A, marker="o", color="tab:blue", label="Good Proxy")
plt.plot(
    units_per_arm_values,
    target_B,
    marker="s",
    color="tab:orange",
    linestyle="--",
    alpha=0.40,
)
plt.plot(units_per_arm_values, cv_B, marker="s", color="tab:orange", label="Bad Proxy")

plt.xlabel("Units per Treatment Arm")
plt.ylabel("Estimated Cumulative Reward")
plt.title("Cross-Validation Estimator")
plt.grid(True)
plt.ylim([-0.001, 0.004])
plt.legend()

plt.tight_layout()
plt.savefig("figs/figure-3.png")
