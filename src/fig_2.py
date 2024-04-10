import numpy as np

from .plotting import (
    get_first_eigenvector,
    plot_simple_multivariate_normal_contour,
    inv_sqrtm,
    get_ols,
)
from .fig_1 import LAMBDA, OMEGA


if __name__ == "__main__":
    target = inv_sqrtm(OMEGA).T @ LAMBDA @ inv_sqrtm(OMEGA)
    plim = inv_sqrtm(OMEGA).T @ (LAMBDA + 2 * np.array(OMEGA) / 100) @ inv_sqrtm(OMEGA)

    plot_simple_multivariate_normal_contour(
        plim,
        0.5,
        "Covariance After Transformation\n(n=100)",
        [
            (get_first_eigenvector(target), "white"),
            ((0.85, 0.85 * get_ols(target)), "blue"),
        ],
        [
            (get_first_eigenvector(plim), "gray"),
            ((0.85, 0.85 * get_ols(plim)), "orange"),
        ],
        fname="figs/figure-2-1.png",
    )

    target = inv_sqrtm(OMEGA).T @ LAMBDA @ inv_sqrtm(OMEGA)
    plim = (
        inv_sqrtm(OMEGA).T @ (LAMBDA + 2 * np.array(OMEGA) / 10000) @ inv_sqrtm(OMEGA)
    )

    plot_simple_multivariate_normal_contour(
        plim,
        0.5,
        "Covariance After Transformation\n(n=10000)",
        [
            (get_first_eigenvector(target), "white"),
            ((0.85, 0.85 * get_ols(target)), "blue"),
        ],
        [
            (get_first_eigenvector(plim), "gray"),
            ((0.85, 0.85 * get_ols(plim)), "orange"),
        ],
        fname="figs/figure-2-2.png",
    )
