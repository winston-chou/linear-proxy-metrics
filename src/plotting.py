import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, sqrtm

inv_sqrtm = lambda matrix: inv(sqrtm(matrix))


def get_coords(bound, num=100):
    """Grid of 2d coordinates to evaluate."""
    assert bound > 0, "Bound must be positive."
    x, y = np.meshgrid(np.linspace(-bound, bound, num), np.linspace(-bound, bound, num))
    coords = np.empty(x.shape + (2,))
    coords[:, :, 0] = x
    coords[:, :, 1] = y
    return coords, x, y


def get_multivariate_normal_pdf(coords, cov_matrix, mean=[0, 0]):
    """Compute multivariate normal PDF of coordinates."""
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    Z = np.empty(coords.shape[:2])
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            diff = coords[i, j] - mean
            exponent = -0.5 * np.dot(np.dot(diff, inv_cov_matrix), diff.T)
            Z[i, j] = np.exp(exponent) / (
                2 * np.pi * np.sqrt(np.linalg.det(cov_matrix))
            )
    return Z


def get_first_eigenvector(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors[:, np.argsort(eigenvalues)[-1]]


def plot_arrow(point, color):
    plt.quiver(
        0,
        0,
        point[0],
        point[1],
        angles="xy",
        scale_units="xy",
        scale=2,
        color=color,
    )


def plot_line(point, color="orange"):
    return plt.plot(
        [-point[0], 0, point[0]],
        [-point[1], 0, point[1]],
        color=color,
        linestyle="--",
    )


def plot_simple_multivariate_normal_contour(
    cov_matrix,
    bound,
    title,
    arrow_configs=[],
    line_configs=[],
    fname=None,
):
    coords, x, y = get_coords(bound)
    Z = get_multivariate_normal_pdf(coords, cov_matrix)

    # Create a contour plot
    plt.figure(figsize=(6, 6), dpi=300)
    plt.contourf(x, y, Z, levels=20, cmap="viridis")

    if arrow_configs:
        for arrow_config, color in arrow_configs:
            plot_arrow(arrow_config, color)
    if line_configs:
        for line_config, color in line_configs:
            plot_line(line_config, color)

    plt.xlim([-bound, bound])
    plt.ylim([-bound, bound])
    plt.xlabel("S")
    plt.ylabel("Y")
    plt.title(
        title
        or "Contour Plot of Multivariate Normal Distribution\nwith First Eigenvector"
    )
    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def get_ols(matrix):
    return matrix[0, 1] / matrix[1, 1]
