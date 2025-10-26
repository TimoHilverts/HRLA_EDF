from ..Algorithm import Algorithm
import numpy as np

class HRLA(Algorithm):
    def __init__(self, d, M, N, K, h, title, U, dU, initial, b=10, alpha=1, beta=1):
        print(f"HRLA instantiating {title}")
        self.b = b
        self.alpha = alpha
        self.beta = beta
        super().__init__(d, M, N, K, h, title, U, dU, initial)

    def _iterate(self, x0, y0, a):
        # Compute parameters of algorithms
        delta   = self.h
        alpha   = self.alpha
        beta    = self.beta
        b       = self.b
        gamma   = a / b
        sigx2   = beta / a
        sigy2   = alpha / b

        # Pre-compute values, to avoid repeated computations
        e       = np.exp(-alpha*delta)
        e2      = np.exp(-2*alpha*delta)
        dU0 = self.dU(x0)

        # Compute mean matrix
        mean_x = x0 - beta * delta * dU0 + (1 - e) / alpha * y0 - gamma / alpha * (delta - (1 - e) / alpha) * dU0
        mean_y = e * y0 - gamma / alpha * (1 - e) * dU0
        mean_matrix = np.block([mean_x, mean_y])

        # Compute covariance matrix
        cov_xx = (2 * sigx2 * delta + sigy2 / alpha ** 3 * (2 * alpha * delta + 1 - e2 - 4 * (1 - e))) * np.eye(self.d)
        cov_yy = sigy2 * (1 - e2) / alpha * np.eye(self.d)
        cov_xy = sigy2 * (1 - e) ** 2 / alpha ** 2 * np.eye(self.d)
        cov_matrix = np.block([[cov_xx, cov_xy], [cov_xy, cov_yy]])

        # Sample new point
        znew = np.random.multivariate_normal(mean_matrix.astype(float), cov_matrix.astype(float))

        return znew[:self.d], znew[self.d:]

