import numpy as np
import sympy as sp
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor, Comparator
import matplotlib.pyplot as plt 
import os

N = 2  #number of contracts
S = 3  #number of customer segments
D = 2  #time slots (day/night)

beta = 0.5

Es = np.array([
    [10, 120],
    [130, 10],
    [50, 50]
])
outside = np.array([6, 1])

tmin, tmax = 0.0, 40.0

#now, we make symbolic notation of the 4 components in both contracts,
#to extend to a d=4 dimensional problem, such that for contract 1, both 
#day/night prices vary, and same for contract 2
l11, l12, l21, l22 = sp.symbols("l11 l12 l21 l22", real=True)

lambda0 = sp.Matrix(outside)  #the static competitor          
lambda1=sp.Matrix([l11,l12]) #contract 1, with l11 price for day, and l12 price at night
lambda2=sp.Matrix([l21,l22]) #contract 2, with l21 price for day, and l22 price at night

lambdas = [lambda0, lambda1, lambda2]

#Below we determine the revenue as a def using sympy. In the shifted_exp part, we subtract -max_c from the exponent part.
#Reason for this is that this stabilizes the problem. It prevents both underflow and overflow.
#e^(-beta*C) -> 0 for large C, or -> 0 for small C, such that it might become undefined in later stages, which this normalization prevents.
def symbolic_revenue(beta):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        costs = [E_s.dot(lam) for lam in lambdas] 
        max_c = sp.Max(*[-beta * c for c in costs])
        shifted_exp = [sp.exp(-beta * c - max_c) for c in costs]
        denom = sum(shifted_exp)
        probs = [e / denom for e in shifted_exp]

        #exclude the outside option
        total += sum(costs[i] * probs[i] for i in range(1, N+1))
    return total

#define U_sym as -revenue, needed for finding the maximum revenue
revenue_sym = symbolic_revenue(beta)
U_sym = -revenue_sym  
gradU_sym = [sp.diff(U_sym, v) for v in (l11, l12, l21, l22)]

#reformulating U and dU as a lambdify function, converting it back to numpy notation
U_func = sp.lambdify((l11, l12, l21, l22), U_sym, "numpy")
gradU_func = sp.lambdify((l11, l12, l21, l22), gradU_sym, "numpy")

def U(x):
    #Here, we now have x = [l11, l12, l21, l22]!!
    vals = np.clip(x, tmin, tmax) #to make sure x is in between tmin=0 and tmax=40
    return float(U_func(*vals))

def dU(x):
    vals = np.clip(x, tmin, tmax)
    grad_vals = np.array(gradU_func(*vals), dtype=float).flatten()
    return grad_vals


d = 4 #number of optimization variables, it tells HRLA how many coordinates lambda_i it needs to update
title = "HRLA_basicmodel_d4"

def initial():
    return np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

algorithm = GO.HRLA(d=d, M=1, N=1, K=150, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithm.generate_samples(As=[1, 2, 3, 4, 5], sim_annealing=False)


#d is the number of optimization variables.
#It tells HRLA how many coordinates lambda_i it needs to update

#M is the number of independent runs. Each run starts from a different random initialization
#For increasing M you can average over runs (improves confidence)

#N is number of samples per iteration per run. At each iteration k,
#HRLA may draw N parallel samples X_k^n per run

#K is number of iterations. Each step evolves the stochastic diff equation
#Larger K -> more steps, better convergence (samples closer to equilibrium)
#too small K -> samples haven't yet reached stationary distribution

#h is step size, U(x) is what HRLA minimizes, dU(x) analytic gradient

#As=[...] are the inverse temperature parameters a.
#Large a -> more concentrated around minima

#sim_annealing: When False, a stays fixed per run.
#When True: a increases with iterations (can accelerate convergence when a is large)

postprocessor = PostProcessor(samples_filename_HRLA)
postprocessor.compute_tables([10, 100, 150], 1, "best") #This provides the tables with a and K vals, together with optimal revenue

#plot the empirical probability distributions for all four components l_ij
#postprocessor.plot_empirical_probabilities(layout="22", tols=[1,2,3,4], dpi=1, running=True)

#With this call (the get_best call), we find for every value for a the best choices for the parameters l_ij, which achieve
#the optimal revenue.
bests = postprocessor.get_best(measured=[10,100,150], dpi=1)

# Automatically choose the a-value that gives the highest revenue

# Compute revenue for each best solution
revenues_best = [-U(np.array(x)) for x in bests]

# Find index of highest revenue
best_idx = int(np.argmax(revenues_best))

best_a = postprocessor.As[best_idx]     # e.g. 3 or 4 or 5
x_opt = np.array(bests[best_idx])       # corresponding optimal lambda values

print(f"Best a = {best_a}")
print(f"Optimal x* = {x_opt}")

# 1D Cut for the best a

tvals = np.linspace(tmin, tmax, 400)
revenues = []

for t in tvals:
    x_temp = x_opt.copy()
    x_temp[2] = t      # vary only l21
    revenues.append(-U(x_temp))

revenues = np.array(revenues)

# Maximum in the 1D cut
idx_max = np.argmax(revenues)
t_max = tvals[idx_max]
R_max = revenues[idx_max]

# Plot

plt.figure(figsize=(8,5))
plt.plot(tvals, revenues, color="black", label="Revenue along 1D cut")
plt.axvline(x=x_opt[2], color="red", linestyle="--",
            label=f"HRLA optimum l21* = {x_opt[2]:.2f}")
plt.axvline(x=t_max, color="green", linestyle="--",
            label=f"1D cut optimum t = {t_max:.2f}")

plt.title(f"1D cut of revenue vs l21 (best a = {best_a})")
plt.xlabel("l21")
plt.ylabel("Revenue")
plt.grid(True)
plt.legend()
plt.tight_layout()

os.makedirs("output/plots", exist_ok=True)
plot_path = f"output/plots/revenue_cut_l21_best_a_{best_a}.png"
plt.savefig(plot_path, dpi=300)

print(f"[SUCCESS] Saved 1D cut for best a ({best_a}) to {plot_path}")

