from ..Algorithm import Algorithm
import numpy as np

class OLA(Algorithm):
    def __init__(self, d, M, N, K, h, title, U, dU, initial, alpha=1):
        print(f"OLA instantiating {title}")
        self.alpha = alpha
        super().__init__(d, M, N, K, h, title, U, dU, initial)

    def _iterate(self, x0, y0, a):
        # Compute parameters of algorithms
        delta   = self.h
        alpha = self.alpha
        sigma2 = alpha / a

        # Pre-compute values, to avoid repeated computations
        dU0 = self.dU(x0)

        # Compute mean matrix
        mean_x = x0 - alpha * delta * dU0

        # Compute covariance matrix
        cov_xx = 2 * sigma2 * delta * np.eye(self.d)

        # Sample new point
        znew = np.random.multivariate_normal(mean_x.astype(float), cov_xx.astype(float))

        return znew, y0
