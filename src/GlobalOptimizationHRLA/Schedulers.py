import numpy as np

class ConstantScheduler:
    """
    Constant temperature scheduler.
    It keeps the inverse temperature parameter constant at a specified value.
    Parameters:
        a (float): The constant value of the inverse temperature.
    """
    def __init__(self, a):
        self.a = a

    def get_inverse_temperature(self, k):
        return self.a

class LinearScheduler:
    """
    Linear temperature scheduler.
    It updates the inverse temperature parameter linearly from start_a to end_a over K steps.
    Parameters:
        start_a (float): Starting value of the inverse temperature.
        end_a (float): Ending value of the inverse temperature.
        K (int): Total number of steps.
    """

    def __init__(self, start_a, end_a, K):
        self.start_a = start_a
        self.end_a = end_a
        self.K = K

    def get_inverse_temperature(self, k):
        return (self.end_a - self.start_a) / self.K * k + self.start_a

class LogScheduler:
    def __init__(self, a_min: float, a_max: float, K: int):
        self.a_min = a_min
        self.a_max = a_max
        self.K = K

    def get_inverse_temperature(self, k: int) -> float:
        return self.a_min + (self.a_max - self.a_min) * (
            np.log(1 + k) / np.log(1 + self.K)
        )

    
class TwoPhaseScheduler:
    def __init__(self, a_min, a_cross, a_max, K, split=0.7):
        self.a_min = a_min
        self.a_cross = a_cross  # e.g. 0.01
        self.a_max = a_max
        self.K = K
        self.K1 = int(split * K)  # 70% of iterations for exploration
        self.K2 = K - self.K1     # 30% for exploitation

    def get_inverse_temperature(self, k):
        if k < self.K1:
            # Phase 1: linear from a_min to a_cross
            return self.a_min + (self.a_cross - self.a_min) * k / self.K1
        else:
            # Phase 2: linear from a_cross to a_max
            return self.a_cross + (self.a_max - self.a_cross) * (k - self.K1) / self.K2

    # In Schedulers.py
class SlowStartLinearScheduler:
    """
    Spends more iterations at small a using quadratic progression.
    a_k = a_min + (a_max - a_min) * (k/K)^2
    
    Gives ~7746 iterations below a=0.007 vs 600 for linear
    at K=100000, a_min=0.001, a_max=1
    """
    def __init__(self, a_min: float, a_max: float, K: int):
        self.a_min = a_min
        self.a_max = a_max
        self.K = K

    def get_inverse_temperature(self, k: int) -> float:
        progress = (k / max(self.K, 1)) ** 2
        return self.a_min + (self.a_max - self.a_min) * progress

class PowerScheduler:
    """
    Spends more iterations at small a using polynomial progression.
    a_k = a_min + (a_max - a_min) * (k/K)^power
    
    Iterations below a=0.007 at K=100000, a_min=0.001, a_max=0.1:
    power=2 (quadratic): 24,618
    power=3 (cubic):     39,300
    power=4 (quartic):   49,600
    """
    def __init__(self, a_min: float, a_max: float, K: int, power: int = 2):
        self.a_min = a_min
        self.a_max = a_max
        self.K = K
        self.power = power

    def get_inverse_temperature(self, k: int) -> float:
        progress = (k / max(self.K, 1)) ** self.power
        return self.a_min + (self.a_max - self.a_min) * progress