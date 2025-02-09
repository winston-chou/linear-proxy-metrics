import numpy as np


def simulate_true_treatment_effects(
    number_of_experiments: int,
    corr: float,
    reward_scale: float,
    proxy_scale: float,
    seed: int,
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    covar = corr * reward_scale * proxy_scale
    Lambda = np.array([[reward_scale**2, covar], [covar, proxy_scale**2]])

    return np.random.multivariate_normal(
        mean=[0, 0], cov=Lambda, size=number_of_experiments
    )


def simulate_observed_treatment_effects(
    true_treatment_effects: np.ndarray,
    units_per_arm: int,
    reward_noise_scale: float,
    proxy_noise_scale: float,
    corr: float,
    folds: int,
) -> np.ndarray:
    if not isinstance(true_treatment_effects, np.ndarray):
        raise TypeError("true_treatment_effects must be a NumPy array.")

    if true_treatment_effects.ndim != 2 or true_treatment_effects.shape[1] != 2:
        raise ValueError("true_treatment_effects must be a (N, 2) array.")

    if not isinstance(folds, int) or folds < 1:
        raise ValueError("folds must be a positive integer.")

    N = true_treatment_effects.shape[0]

    noise_covar = reward_noise_scale * proxy_noise_scale * corr
    Omega = np.array(
        [[reward_noise_scale**2, noise_covar], [noise_covar, proxy_noise_scale**2]]
    )

    if folds == 1:

        scaled_covar = Omega * 2 / units_per_arm
        noise = np.random.multivariate_normal(mean=[0, 0], cov=scaled_covar, size=N)
        observed_treatment_effects = true_treatment_effects + noise
    else:

        scaled_covar = Omega * 2 * folds / units_per_arm

        noise = np.random.multivariate_normal(
            mean=[0, 0], cov=scaled_covar, size=(N, folds)
        )

        noise = noise.transpose(0, 2, 1)

        observed_treatment_effects = true_treatment_effects[:, :, np.newaxis] + noise

    return observed_treatment_effects


def estimate_treatment_sums(
    true_treatment_effects: np.ndarray, observed_treatment_effects: np.ndarray
) -> dict:
    if not isinstance(true_treatment_effects, np.ndarray):
        raise TypeError("true_treatment_effects must be a NumPy array.")

    if not isinstance(observed_treatment_effects, np.ndarray):
        raise TypeError("observed_treatment_effects must be a NumPy array.")

    if true_treatment_effects.ndim != 2 or true_treatment_effects.shape[1] != 2:
        raise ValueError("true_treatment_effects must be a (N, 2) array.")

    N = true_treatment_effects.shape[0]
    if observed_treatment_effects.shape[0] != N:
        raise ValueError(
            "Number of experiments (N) must match between true and observed treatment effects."
        )

    true_R = true_treatment_effects[:, 0]

    if (
        observed_treatment_effects.ndim == 2
        and observed_treatment_effects.shape[1] == 2
    ):

        observed_R = observed_treatment_effects[:, 0]
        observed_S = observed_treatment_effects[:, 1]

        mask_observed_S_positive = observed_S > 0
        sum_true_R_if_observed_S_positive = np.sum(true_R[mask_observed_S_positive])

        sum_observed_R_if_observed_S_positive = np.sum(
            observed_R[mask_observed_S_positive]
        )

        mask_observed_R_positive = observed_R > 0
        sum_true_R_if_observed_R_positive = np.sum(true_R[mask_observed_R_positive])

        mask_true_R_positive = true_R > 0
        sum_true_R_if_true_R_positive = np.sum(true_R[mask_true_R_positive])

        sum_observed_R_if_observed_R_positive = np.sum(
            observed_R[mask_observed_R_positive]
        )

        sum_cross_fold_R_if_observed_S_positive = None

        denominator = np.sum(mask_true_R_positive)
        if denominator > 0:
            numerator = np.sum(mask_observed_S_positive & mask_true_R_positive)
            prob_observed_S_positive_given_true_R_positive = numerator / denominator
        else:
            prob_observed_S_positive_given_true_R_positive = np.nan

        prob_mean_S_other_folds_positive_given_true_R_positive = None

    elif (
        observed_treatment_effects.ndim == 3
        and observed_treatment_effects.shape[1] == 2
    ):

        observed_R = observed_treatment_effects[:, 0, :]
        observed_S = observed_treatment_effects[:, 1, :]

        mean_observed_R = observed_R.mean(axis=1)
        mean_observed_S = observed_S.mean(axis=1)

        folds = observed_treatment_effects.shape[2]

        mask_mean_observed_S_positive = mean_observed_S > 0
        sum_true_R_if_observed_S_positive = np.sum(
            true_R[mask_mean_observed_S_positive]
        )

        sum_observed_R_if_observed_S_positive = np.sum(
            mean_observed_R[mask_mean_observed_S_positive]
        )

        mask_mean_observed_R_positive = mean_observed_R > 0
        sum_true_R_if_observed_R_positive = np.sum(
            true_R[mask_mean_observed_R_positive]
        )

        mask_true_R_positive = true_R > 0
        sum_true_R_if_true_R_positive = np.sum(true_R[mask_true_R_positive])

        sum_observed_R_if_observed_R_positive = np.sum(
            mean_observed_R[mask_mean_observed_R_positive]
        )

        total_S = observed_S.sum(axis=1)

        sum_S_other_folds = total_S[:, np.newaxis] - observed_S

        mean_S_other_folds = sum_S_other_folds / (folds - 1)

        mask_mean_S_other_folds_positive = mean_S_other_folds > 0

        sum_cross_fold_R_if_observed_S_positive = np.sum(
            observed_R[mask_mean_S_other_folds_positive]
        )

        sum_cross_fold_R_if_observed_S_positive /= folds

        denominator = np.sum(mask_true_R_positive)
        if denominator > 0:
            numerator = np.sum(mask_mean_observed_S_positive & mask_true_R_positive)
            prob_observed_S_positive_given_true_R_positive = numerator / denominator
        else:
            prob_observed_S_positive_given_true_R_positive = np.nan

        prob_per_fold = np.empty(folds)
        for f in range(folds):
            mask_S_other_positive = mean_S_other_folds[:, f] > 0
            numerator_f = np.sum(mask_S_other_positive & mask_true_R_positive)
            if denominator > 0:
                prob_per_fold[f] = numerator_f / denominator
            else:
                prob_per_fold[f] = np.nan

        prob_mean_S_other_folds_positive_given_true_R_positive = np.nanmean(
            prob_per_fold
        )

    else:
        raise ValueError(
            "observed_treatment_effects must be either (N, 2) or (N, 2, folds) array."
        )

    return {
        "sum_true_R_if_observed_S_positive": sum_true_R_if_observed_S_positive,
        "sum_observed_R_if_observed_S_positive": sum_observed_R_if_observed_S_positive,
        "sum_true_R_if_observed_R_positive": sum_true_R_if_observed_R_positive,
        "sum_true_R_if_true_R_positive": sum_true_R_if_true_R_positive,
        "sum_observed_R_if_observed_R_positive": sum_observed_R_if_observed_R_positive,
        "sum_cross_fold_R_if_observed_S_positive": sum_cross_fold_R_if_observed_S_positive,
        "prob_observed_S_positive_given_true_R_positive": prob_observed_S_positive_given_true_R_positive,
        "prob_mean_S_other_folds_positive_given_true_R_positive": prob_mean_S_other_folds_positive_given_true_R_positive,
    }
