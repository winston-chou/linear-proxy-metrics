import numpy as np
from .plotting import (
    get_first_eigenvector,
    plot_simple_multivariate_normal_contour,
)


LAMBDA = np.array([[1, -0.25], [-0.25, 1]]) / 1000
OMEGA = np.array([[1, 0.4], [0.4, 1]])


if __name__ == "__main__":
    plot_simple_multivariate_normal_contour(
        LAMBDA,
        0.5,
        "True Treatment Effect Covariance",
        arrow_configs=[(get_first_eigenvector(LAMBDA), "white")],
        fname="figs/figure-1-1.png",
    )

    plot_simple_multivariate_normal_contour(
        OMEGA,
        0.5,
        "Unit-Level Sampling Covariance",
        arrow_configs=[
            (get_first_eigenvector(LAMBDA), "white"),
            (get_first_eigenvector(OMEGA), "black"),
        ],
        fname="figs/figure-1-2.png",
    )

    plot_simple_multivariate_normal_contour(
        LAMBDA + 2 * np.array(OMEGA) / 3200,
        0.5,
        "Estimated Treatment Effect Covariance",
        arrow_configs=[
            (get_first_eigenvector(LAMBDA), "white"),
            (get_first_eigenvector(OMEGA), "black"),
            (
                (1, 0),
                "blue",
            ),  # We fix an eigenvector as both eigenvalues are identical.
        ],
        fname="figs/figure-1-3.png",
    )
