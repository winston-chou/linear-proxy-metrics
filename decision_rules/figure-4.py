from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


from constants import (
    DEFAULT_UNITS_PER_ARM,
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


N_values = [20, 50, 100, 200, 500, 1000]


def run_scenario(param_dict, N, units_per_arm, num_sims):
    """
    For a given scenario's param_dict and a given N,
    run 'num_sims' random simulations and return:
      - The approximate "target" (closed-form)
      - Arrays of naive sums
      - Arrays of CV sums
    """

    target = compute_expected_sum_true_R_if_observed_S_positive(
        N=N,
        corr_true=param_dict["corr_true"],
        reward_scale=param_dict["reward_scale"],
        proxy_scale=param_dict["proxy_scale"],
        corr_observed=param_dict["corr_observed"],
        units_per_arm=units_per_arm,
        reward_noise_scale=param_dict["reward_noise_scale"],
        proxy_noise_scale=param_dict["proxy_noise_scale"],
        folds=param_dict["folds"],
    )

    naive_sums = []
    cv_sums = []

    for sim in tqdm(range(num_sims)):

        true_effects = simulate_true_treatment_effects(
            number_of_experiments=N,
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

    naive_sums = np.array(naive_sums)
    cv_sums = np.array(cv_sums)

    return target, naive_sums, cv_sums


scenario_A_results = []
scenario_B_results = []

for N_ in N_values:

    target_A, naive_A_sims, cv_A_sims = run_scenario(
        params_A, N_, DEFAULT_UNITS_PER_ARM, NUM_SIMULATIONS
    )

    target_B, naive_B_sims, cv_B_sims = run_scenario(
        params_B, N_, DEFAULT_UNITS_PER_ARM, NUM_SIMULATIONS
    )

    scenario_A_results.append((N_, target_A, naive_A_sims, cv_A_sims))
    scenario_B_results.append((N_, target_B, naive_B_sims, cv_B_sims))


def mean_and_CI(values):
    """
    Returns (mean, lower_CI, upper_CI) for ~95% CI using +/-1.96 * std / sqrt(n).
    """
    mean_val = np.mean(values)
    stderr = np.std(values, ddof=1) / sqrt(len(values))
    ci_radius = 1.96 * stderr
    return mean_val, mean_val - ci_radius, mean_val + ci_radius


N_vals_A = []
naive_mean_A = []
naive_low_A = []
naive_high_A = []
cv_mean_A = []
cv_low_A = []
cv_high_A = []
target_vals_A = []

for (N_, target_A, naive_A_sims, cv_A_sims) in scenario_A_results:
    N_vals_A.append(N_)

    mean_naive_A, low_naive_A, high_naive_A = mean_and_CI(naive_A_sims)
    naive_mean_A.append(mean_naive_A)
    naive_low_A.append(low_naive_A)
    naive_high_A.append(high_naive_A)

    mean_cv_A, low_cv_A, high_cv_A = mean_and_CI(cv_A_sims)
    cv_mean_A.append(mean_cv_A)
    cv_low_A.append(low_cv_A)
    cv_high_A.append(high_cv_A)

    target_vals_A.append(target_A)


N_vals_B = []
naive_mean_B = []
naive_low_B = []
naive_high_B = []
cv_mean_B = []
cv_low_B = []
cv_high_B = []
target_vals_B = []

for (N_, target_B, naive_B_sims, cv_B_sims) in scenario_B_results:
    N_vals_B.append(N_)

    mean_naive_B, low_naive_B, high_naive_B = mean_and_CI(naive_B_sims)
    naive_mean_B.append(mean_naive_B)
    naive_low_B.append(low_naive_B)
    naive_high_B.append(high_naive_B)

    mean_cv_B, low_cv_B, high_cv_B = mean_and_CI(cv_B_sims)
    cv_mean_B.append(mean_cv_B)
    cv_low_B.append(low_cv_B)
    cv_high_B.append(high_cv_B)

    target_vals_B.append(target_B)


N_vals_A = np.array(N_vals_A)
N_vals_B = np.array(N_vals_B)

naive_mean_A = np.array(naive_mean_A)
naive_low_A = np.array(naive_low_A)
naive_high_A = np.array(naive_high_A)

cv_mean_A = np.array(cv_mean_A)
cv_low_A = np.array(cv_low_A)
cv_high_A = np.array(cv_high_A)

target_vals_A = np.array(target_vals_A)

naive_mean_B = np.array(naive_mean_B)
naive_low_B = np.array(naive_low_B)
naive_high_B = np.array(naive_high_B)

cv_mean_B = np.array(cv_mean_B)
cv_low_B = np.array(cv_low_B)
cv_high_B = np.array(cv_high_B)

target_vals_B = np.array(target_vals_B)


fig, (ax_true, ax_naive, ax_cv) = plt.subplots(3, 1, figsize=(5, 9), sharey=True)


ax_true.set_title("True Cumulative Reward")
ax_true.set_xlabel("Number of Experiments")
ax_true.set_ylabel("True Cumulative Reward")


ax_true.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_true.plot(N_vals_A, target_vals_A, "o-", label="Good Proxy", color="tab:blue")


ax_true.plot(N_vals_B, target_vals_B, "s-", label="Bad Proxy", color="tab:orange")

ax_true.grid(True)
ax_true.legend()


ax_naive.set_title("Naive Estimator")
ax_naive.set_xlabel("Number of Experiments")
ax_naive.set_ylabel("Estimated Cumulative Reward")


ax_naive.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_naive.plot(N_vals_A, naive_mean_A, "o-", label="Good Proxy", color="tab:blue")
ax_naive.plot(N_vals_A, target_vals_A, "o--", alpha=0.40, color="tab:blue")


ax_naive.plot(N_vals_B, naive_mean_B, "s-", label="Bad Proxy", color="tab:orange")
ax_naive.plot(N_vals_B, target_vals_B, "s--", alpha=0.40, color="tab:orange")

ax_naive.grid(True)
ax_naive.legend()


ax_cv.set_title("CV Estimator")
ax_cv.set_xlabel("Number of Experiments")
ax_cv.set_ylabel("Estimated Cumulative Reward")


ax_cv.axhline(0, color="k", linestyle="--", linewidth=0.8)
ax_cv.plot(N_vals_A, cv_mean_A, "o-", label="Good Proxy", color="tab:blue")
ax_cv.plot(N_vals_A, target_vals_A, "o--", alpha=0.40, color="tab:blue")


ax_cv.plot(N_vals_B, cv_mean_B, "s-", label="Bad Proxy", color="tab:orange")
ax_cv.plot(N_vals_B, target_vals_B, "s--", alpha=0.40, color="tab:orange")

ax_cv.grid(True)
ax_cv.legend()

plt.tight_layout()
plt.savefig("figs/figure-4.png")
