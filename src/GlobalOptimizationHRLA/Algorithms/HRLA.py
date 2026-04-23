from ..Algorithm import Algorithm
import numpy as np

class HRLA(Algorithm):
    def __init__(self, d, M, N, K, h, title, U, dU, initial, b=10, alpha=1, beta=1, a_min=0.1):
        self.b = b
        self.alpha = alpha
        self.beta = beta
        super().__init__(d, M, N, K, h, title, U, dU, initial, a_min=a_min, power=power)

    def get_step_size(self, k):
        if callable(self.h):
            return self.h(k, self.K)
        return self.h

    def _iterate(self, x0, y0, a, k):
        delta   = self.get_step_size(k)
        alpha   = self.alpha
        beta    = self.beta
        b       = self.b
        gamma   = a / b
        sigx2   = beta / a
        sigy2   = alpha / b

        e       = np.exp(-alpha * delta)
        e2      = np.exp(-2 * alpha * delta)
        dU0     = self.dU(x0)

        mean_x = x0 - beta * delta * dU0 + (1 - e) / alpha * y0 - gamma / alpha * (delta - (1 - e) / alpha) * dU0
        mean_y = e * y0 - gamma / alpha * (1 - e) * dU0
        mean_matrix = np.block([mean_x, mean_y])

        cov_xx = (2 * sigx2 * delta + sigy2 / alpha**3 * (2 * alpha * delta + 1 - e2 - 4 * (1 - e))) * np.eye(self.d)
        cov_yy = sigy2 * (1 - e2) / alpha * np.eye(self.d)
        cov_xy = sigy2 * (1 - e)**2 / alpha**2 * np.eye(self.d)
        cov_matrix = np.block([[cov_xx, cov_xy], [cov_xy, cov_yy]])

        znew = np.random.multivariate_normal(mean_matrix.astype(float), cov_matrix.astype(float))
        return znew[:self.d], znew[self.d:]