import time
import os
import numpy as np
from tqdm import tqdm
import multiprocess as mp
import dill

from .Schedulers import ConstantScheduler, LinearScheduler


class Algorithm:
    def __init__(self, d, M, N, K, h, title, U, dU, initial):
        print(f"Initializing Algorithm for {title} with d={d}, M={M}, N={N}, K={K}, h={h}")
        self.d = d
        self.M = M
        self.N = N
        self.K = K
        self.h = h

        self.title = title
        self.U = U
        self.dU = dU
        self.initial = initial

    def __get_samples(self, a, sim_annealing, seed=None):
        # Set random seed if specified
        if seed: np.random.seed(seed)

        # Empty array to store iterates
        samples = np.zeros((self.N, self.K, self.d))

        # Initialize the step-size scheduler
        scheduler = LinearScheduler(0.1, a, self.K) if sim_annealing else ConstantScheduler(a)

        # Generate N samples
        for n in range(self.N):
            # Initial sample
            x0, y0 = self.initial(), np.zeros(self.d)

            # Perform K iterations
            for k in range(self.K):
                # Determine the inverse temperature parameter
                ak = scheduler.get_inverse_temperature(k)

                # Perform iteration
                x0, y0 = self._iterate(x0, y0, ak)

                # Store marginal sample
                samples[n, k] = x0

        return samples

    def generate_samples(self, As, sim_annealing):
        print(f"Generating samples for a in {As} with sim_annealing={sim_annealing}")

        # Function to generate samples
        generate_task = lambda task_nb: np.array([self.__get_samples(a, sim_annealing, seed=42 + task_nb) for a in As])

        # Generate samples in parallel
        with mp.Pool() as pool:
            samples = list(tqdm(pool.imap(generate_task, range(self.M)), total=self.M, desc="Generating Samples"))

        # Format the data to save it
        samples_filename = f'temp_output/data/{self.title}_{time.time()}.pickle'
        samples_data = {
            "samples": samples,
            "title": self.title,
            "U": self.U,
            "dU": self.dU,
            "intial": self.initial,
            "d": self.d,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "h": self.h,
            "As": As,
            "sim_annealing": sim_annealing,
        }

        # Save all the data
        if not os.path.exists("temp_output"):
            os.makedirs("temp_output")
        if not os.path.exists("temp_output/data"):
            os.makedirs("temp_output/data")
        with open(samples_filename, 'wb') as handle:
            dill.dump(samples_data, handle)
            print(f"[SUCCESS] Samples dumped successfully into file {samples_filename}")

        return samples_filename

    def _iterate(self, x0, y0, a):
        raise NotImplementedError("The iterate method must be implemented in the child class")
