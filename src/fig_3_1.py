from multiprocessing import Pool

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .simulation import get_estimates
from .constants import SEED, SAMPLE_SIZES, OMEGA, LAMBDA, BETA, SIMS


RNG = np.random.RandomState(seed=SEED)


def get_estimates_partial(i):
    return get_estimates(i, SAMPLE_SIZES, RNG, OMEGA, LAMBDA, surrogacy_params=BETA)


if __name__ == "__main__":
    iterations = list(range(SIMS))

    with Pool() as pool:
        estimates = pool.map(get_estimates_partial, iterations)

    estimates_df = pd.concat(estimates)

    def _agg(dataframe):
        return pd.Series(
            {
                "bias_beta_1": dataframe.bias_beta_1.mean(),
                "bias_beta_2": dataframe.bias_beta_2.mean(),
                "std_beta_1": np.sqrt(dataframe.beta_1.var()),
                "std_beta_2": np.sqrt(dataframe.beta_2.var()),
                "abs_bias_beta_1": np.abs(dataframe.bias_beta_1).mean(),
                "abs_bias_beta_2": np.abs(dataframe.bias_beta_2).mean(),
                "rmse_beta_1": np.sqrt((dataframe.bias_beta_1**2).mean()),
                "rmse_beta_2": np.sqrt((dataframe.bias_beta_2**2).mean()),
            }
        )

    stats = (
        estimates_df.assign(
            bias_beta_1=lambda df: df.beta_1 - BETA[0],
            bias_beta_2=lambda df: df.beta_2 - BETA[1],
        )
        .groupby(["sample_size", "estimator"])
        .apply(_agg)
        .reset_index()
        .query("estimator != 'truth'")
    )

    fig, axes = plt.subplots(2, 3, figsize=[8, 4], dpi=200)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    ax1.plot(
        stats[stats.estimator == "naive"].sample_size,
        stats[stats.estimator == "naive"].bias_beta_1,
        label="naive",
    )
    ax1.plot(
        stats[stats.estimator == "liml"].sample_size,
        stats[stats.estimator == "liml"].bias_beta_1,
        label="limlk",
        linestyle=":",
    )
    ax1.plot(
        stats[stats.estimator == "tc"].sample_size,
        stats[stats.estimator == "tc"].bias_beta_1,
        linestyle="--",
        label="tc",
    )
    ax1.set_xlabel("n")
    ax1.set_ylabel(r"Bias of $\hat{\beta}_1$")
    ax1.plot(SAMPLE_SIZES, [0] * len(SAMPLE_SIZES), color="lightgray", zorder=0)
    ax1.legend()

    ax2.plot(
        stats[stats.estimator == "naive"].sample_size,
        stats[stats.estimator == "naive"].rmse_beta_1,
        label="naive",
    )
    ax2.plot(
        stats[stats.estimator == "liml"].sample_size,
        stats[stats.estimator == "liml"].rmse_beta_1,
        label="limlk",
        linestyle=":",
    )
    ax2.plot(
        stats[stats.estimator == "tc"].sample_size,
        stats[stats.estimator == "tc"].rmse_beta_1,
        linestyle="--",
        label="tc",
    )
    ax2.set_xlabel("n")
    ax2.set_ylabel(r"RMSE of $\hat{\beta}_1$")
    ax2.plot(SAMPLE_SIZES, [0] * len(SAMPLE_SIZES), color="lightgray", zorder=0)
    ax2.legend()

    ax3.plot(
        stats[stats.estimator == "naive"].sample_size,
        stats[stats.estimator == "naive"].std_beta_1,
        label="naive",
    )
    ax3.plot(
        stats[stats.estimator == "liml"].sample_size,
        stats[stats.estimator == "liml"].std_beta_1,
        label="limlk",
        linestyle=":",
    )
    ax3.plot(
        stats[stats.estimator == "tc"].sample_size,
        stats[stats.estimator == "tc"].std_beta_1,
        linestyle="--",
        label="tc",
    )
    ax3.set_xlabel("n")
    ax3.set_ylabel(r"Std. Dev. of $\hat{\beta}_1$")
    ax3.plot(SAMPLE_SIZES, [0] * len(SAMPLE_SIZES), color="lightgray", zorder=0)
    ax3.legend()

    ax4.plot(
        stats[stats.estimator == "naive"].sample_size,
        stats[stats.estimator == "naive"].bias_beta_2,
        label="naive",
    )
    ax4.plot(
        stats[stats.estimator == "liml"].sample_size,
        stats[stats.estimator == "liml"].bias_beta_2,
        label="limlk",
        linestyle=":",
    )
    ax4.plot(
        stats[stats.estimator == "tc"].sample_size,
        stats[stats.estimator == "tc"].bias_beta_2,
        linestyle="--",
        label="tc",
    )
    ax4.set_xlabel("n")
    ax4.set_ylabel(r"Bias of $\hat{\beta}_2$")
    ax4.plot(SAMPLE_SIZES, [0] * len(SAMPLE_SIZES), color="lightgray", zorder=0)
    ax4.legend()

    ax5.plot(
        stats[stats.estimator == "naive"].sample_size,
        stats[stats.estimator == "naive"].rmse_beta_2,
        label="naive",
    )
    ax5.plot(
        stats[stats.estimator == "liml"].sample_size,
        stats[stats.estimator == "liml"].rmse_beta_2,
        label="limlk",
        linestyle=":",
    )
    ax5.plot(
        stats[stats.estimator == "tc"].sample_size,
        stats[stats.estimator == "tc"].rmse_beta_2,
        linestyle="--",
        label="tc",
    )
    ax5.set_xlabel("n")
    ax5.set_ylabel(r"RMSE of $\hat{\beta}_2$")
    ax5.plot(SAMPLE_SIZES, [0] * len(SAMPLE_SIZES), color="lightgray", zorder=0)
    ax5.legend()

    ax6.plot(
        stats[stats.estimator == "naive"].sample_size,
        stats[stats.estimator == "naive"].std_beta_2,
        label="naive",
    )
    ax6.plot(
        stats[stats.estimator == "liml"].sample_size,
        stats[stats.estimator == "liml"].std_beta_2,
        label="limlk",
        linestyle=":",
    )
    ax6.plot(
        stats[stats.estimator == "tc"].sample_size,
        stats[stats.estimator == "tc"].std_beta_2,
        linestyle="--",
        label="tc",
    )
    ax6.set_xlabel("n")
    ax6.set_ylabel(r"Std. Dev. of $\hat{\beta}_2$")
    ax6.plot(SAMPLE_SIZES, [0] * len(SAMPLE_SIZES), color="lightgray", zorder=0)
    ax6.legend()

    plt.suptitle("No Direct Effects")
    fig.tight_layout()
    plt.savefig("figs/figure-3-1.png")
