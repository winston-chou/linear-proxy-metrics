# Simulation params
import numpy as np


SIMS = 100
BETA = np.array([-0.4, 0.04])

LAMBDA = (
    np.array(
        [
            [1, *BETA],
            [BETA[0], 1, 0],
            [BETA[1], 0, 1],
        ]
    )
    / 1000
)

SCALE = np.sqrt(np.array([0.01, 10, 25]))
OMEGA = np.array(
    [
        [1.00, 0.80, 0.00],
        [0.80, 1.00, -0.1],
        [0.00, -0.1, 1.00],
    ]
) * np.outer(SCALE, SCALE)


SAMPLE_SIZES = [5_000, 10_000, 20_000, 50_000, 100_000, 150_000, 200_000]
SEED = 1024
