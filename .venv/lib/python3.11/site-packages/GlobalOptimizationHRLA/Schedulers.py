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
