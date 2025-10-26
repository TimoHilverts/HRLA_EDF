from ..Algorithm import Algorithm
import numpy as np

class ULA(Algorithm):
    def __init__(self, d, M, N, K, h, title, U, dU, initial, b=10, alpha=1):
        print(f"ULA instantiating {title}")
        self.b = b
        self.alpha = alpha
        super().__init__(d, M, N, K, h, title, U, dU, initial)

    def _iterate(self, x0, y0, a):
        # Compute parameters of algorithms
        delta   = self.h

        b = self.b
        alpha = self.alpha
        sigma2 = alpha / b
        gamma = a / b

        v0 = y0

        # Pre-compute values, to avoid repeated computations
        e = np.exp(-alpha * delta)
        dU0 = self.dU(x0)

        # Compute mean matrix
        mean_x = x0 + (1 - e) / alpha * v0 - gamma / alpha * (delta - (1 - e) / alpha) * dU0
        mean_v = e * v0 - gamma / alpha * (1 - e) * dU0
        mean_matrix = np.block([mean_x, mean_v])

        # Compute covariance matrix
        cov_xx = 2 * sigma2 / (alpha ** 2) * (delta - 2 * (1 - e) / alpha + (1 - e ** 2) / (2 * alpha) ) * np.eye(self.d)
        cov_vv = 2 * sigma2 / (2 * alpha) * (1 - e ** 2) * np.eye(self.d)
        cov_xv = 2 * sigma2 / alpha * ((1 - e) / alpha - (1 - e ** 2) / (2 * alpha)) * np.eye(self.d)
        cov_matrix = np.block([[cov_xx, cov_xv], [cov_xv, cov_vv]])

        # Sample new point
        znew = np.random.multivariate_normal(mean_matrix.astype(float), cov_matrix.astype(float))

        return znew[:self.d], znew[self.d:]