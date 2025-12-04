import numpy as np
import sympy as sp
from sympy import Piecewise, Eq
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor, Comparator
import matplotlib.pyplot as plt 
import os

N = 2  #number of contracts
S = 3  #number of customer segments
D = 2  #time slots (day/night)

Es_original = np.array([
    [10, 120],
    [130, 10],
    [50, 50]
])
outside1 = np.array([6, 1])
outside2 = np.array([8, 3])
outside3 = np.array([4, 3])

competitors = [outside1, outside2, outside3] #combining the competitors into one matrix

competitors_sym=[sp.Matrix(c) for c in competitors]

def reservation_bill(E): #This def provides the minimal competitor (outside option), which represents the reservation bill
    comp_costs=[E.dot(c) for c in competitors_sym]
    
    return sp.Min(*comp_costs)

tmin, tmax = 0.0, 40.0

#now, we make symbolic notation of the 4 components in both contracts,
#to extend to a d=4 dimensional problem, such that for contract 1, both 
#day/night prices vary, and same for contract 2
f1, f2, l11, l12, l21, l22 = sp.symbols("f1 f2 l11 l12 l21 l22", real=True)
       
lambda1=sp.Matrix([l11,l12]) #contract 1, with l11 price for day, and l12 price at night
lambda2=sp.Matrix([l21,l22]) #contract 2, with l21 price for day, and l22 price at night

lambdas = [lambda1, lambda2]

def shifting_cons(E, shift):
    #E=[E_day, E_night]
    E_day, E_night = E
    E_day_new = (1-shift)*E_day
    E_night_new = E_night+shift*E_day
    
    return np.array([E_day_new, E_night_new])

shift=0.25
Es=np.array([shifting_cons(E, shift) for E in Es_original])

#Below we determine the revenue as a def using sympy. In the shifted_exp part, we subtract -max_c from the exponent part.
#Reason for this is that this stabilizes the problem. It prevents both underflow and overflow.
#e^(-beta*C) -> 0 for large C, or -> 0 for small C, such that it might become undefined in later stages, which this normalization prevents.
def symbolic_revenue(beta, w):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        R_s=reservation_bill(E_s) #provides the reservation bill, so the minimum of outsiders

        cost1=f1+E_s.dot(lambda1) #represents company cost, looking at contract 1
        cost2=f2+E_s.dot(lambda2) #represents company cost, looking at contract 2

        cost_company=[cost1, cost2] #represents the total cost for the company

        cost_customer=[R_s, cost1, cost2]

        U_1 = cost1 - R_s #this is the disutility, considering cost 1, which is U =C1-Rs. Represents the difference between the 
                            # the first contract and minimum competitor, which we want to minimize of course
        U_2 = cost2 - R_s

        utilities=[0, U_1, U_2] #here, the utility of the competitor is 0

        max_ulitity=sp.Max(0,U_1, U_2) #Needed for stabilizing the problem

        #max_c = sp.Max(*[-beta * c for c in cost_customer]) #here, we include the max_c term to make sure the model remains stable
        exp_term = [sp.exp(-beta * (u - max_ulitity)) for u in utilities]
        #exp_term = [sp.exp(-beta * c - max_c) for c in cost_customer]
        denom = sum(exp_term)
        probs = [e / denom for e in exp_term]

        #exclude the outside option for the revenue
        total += w[s]*sum(cost_company[i] * probs[i+1] for i in range(N))
    return total

w_basic=[1, 1, 1]
w_equal=[1/3, 1/3, 1/3]
#w_diff=[w1, w2, w3]

beta = 0.05 

#define U_sym as -revenue, needed for finding the maximum revenue
revenue_sym = symbolic_revenue(beta, w_basic) #here we now include w_basic=[1, 1, 1] to check the basic case
U_sym = -revenue_sym  

variables=(f1, f2, l11, l12, l21, l22)
gradU_sym = [sp.diff(U_sym, v) for v in variables] #this is the gradient wrt all lambdas

#reformulating U and dU as a lambdify function, converting it back to numpy notation
U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

def U(x):
    #Here, we now have x = [f1, f2, l11, l12, l21, l22]!!
    vals = np.clip(x, tmin, tmax)
    return float(U_func(*vals))

def dU(x):
    vals = np.clip(x, tmin, tmax)
    grad_vals = np.array(gradU_func(*vals), dtype=float).flatten()
    return grad_vals


d = 6 #number of optimization variables, it tells HRLA how many coordinates lambda_i it needs to update
title = "HRLA_basicmodel_d6"

def initial():
    return np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

algorithm = GO.HRLA(d=d, M=1, N=1, K=5000, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithm.generate_samples(As=[5], sim_annealing=True)


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
postprocessor.compute_tables([10, 2500, 5000], 1, "best") #This provides the tables with a and K vals, together with optimal revenue

#plot the empirical probability distributions for all four components l_ij
#postprocessor.plot_empirical_probabilities(layout="22", tols=[1,2,3,4], dpi=1, running=True)

#With this call (the get_best call), we find for every value for a the best choices for the parameters l_ij, which achieve
#the optimal revenue.
bests = postprocessor.get_best(measured=[10, 2500,5000], dpi=10)

# --------------------------------------------------------------------
# Automatically choose the a-value that gives the highest revenue
# --------------------------------------------------------------------
# Compute revenue for each best solution
revenues_best = [-U(np.array(x)) for x in bests]

# Find index of highest revenue
best_idx = int(np.argmax(revenues_best))

best_a = postprocessor.As[best_idx]     # e.g. 3 or 4 or 5
x_opt = np.array(bests[best_idx])       # corresponding optimal f and lambda values

print(f"Best a = {best_a}")
print(f"Optimal x* = {x_opt}")

# --------------------------------------------------------------------
# 1D Cut for the best a
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# 1D CUTS FOR ALL 6 VARIABLES
# --------------------------------------------------------------------
var_names = ["f1", "f2", "l11", "l12", "l21", "l22"]  # names for plots

tvals = np.linspace(tmin, tmax, 400)

os.makedirs("output/plots", exist_ok=True)

for i, name in enumerate(var_names):

    revenues = []
    for t in tvals:
        x_temp = x_opt.copy()
        x_temp[i] = t
        revenues.append(-U(x_temp))

    revenues = np.array(revenues)

    # Find best 1D cut optimum
    idx_max = np.argmax(revenues)
    t_max = tvals[idx_max]
    R_max = revenues[idx_max]

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(tvals, revenues, color="black", label=f"Revenue along 1D cut of {name}")
    plt.axvline(x=x_opt[i], color="red", linestyle="--",
                label=f"HRLA optimum {name}* = {x_opt[i]:.2f}")
    plt.axvline(x=t_max, color="green", linestyle="--",
                label=f"1D cut optimum = {t_max:.2f}")

    plt.title(f"1D cut of revenue vs {name} (best a = {best_a})")
    plt.xlabel(name)
    plt.ylabel("Revenue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save
    plot_path = f"output/plots/1dcut_{name}_BM_complete_a5_best_a_{best_a}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved 1D cut for {name} in BM_complete a5 to {plot_path}")
