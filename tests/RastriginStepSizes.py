import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# Define Rastrigin function, its gradient and an initial distribution
d = 10
title = "Rastrigin"
U = lambda x: d + np.linalg.norm(x) ** 2 - np.sum(np.cos(2*np.pi*x))
dU = lambda x: 2 * x + 2 * np.pi * np.sin(2*np.pi*x)
initial = lambda: np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

# Step-size of h=0.001
algorithm001 = GO.HRLA(d=d, M=100, N=10, K=14000, h=0.001, title=title, U=U, dU=dU, initial=initial)
samples_filename001 = algorithm001.generate_samples(As=[1,2,3,4], sim_annealing=False)
postprocessor001 = PostProcessor(samples_filename001)
postprocessor001.compute_tables([14000], 100, "mean")
postprocessor001.compute_tables([14000], 100, "median")
postprocessor001.compute_tables([14000], 100, "std")

# Step-size of h=0.01
algorithm01 = GO.HRLA(d=d, M=100, N=10, K=14000, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename01 = algorithm01.generate_samples(As=[1,2,3,4], sim_annealing=False)
postprocessor01 = PostProcessor(samples_filename01)
postprocessor01.compute_tables([14000], 100, "mean")
postprocessor01.compute_tables([14000], 100, "median")
postprocessor01.compute_tables([14000], 100, "std")

# Step-size of h=0.1
algorithm1 = GO.HRLA(d=d, M=100, N=10, K=14000, h=0.1, title=title, U=U, dU=dU, initial=initial)
samples_filename1 = algorithm1.generate_samples(As=[1,2,3,4], sim_annealing=False)
postprocessor1 = PostProcessor(samples_filename1)
postprocessor1.compute_tables([14000], 100, "mean")
postprocessor1.compute_tables([14000], 100, "median")
postprocessor1.compute_tables([14000], 100, "std")