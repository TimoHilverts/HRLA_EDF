import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor, Comparator

# --- Basic model constants (as in your plotting snippet) ---
N = 2   ## 2 contracts
S = 3   ## 3 segments
D = 2   ## 2 time slots
Es = np.array([
    [10,120],   ## segment 1 consumes mostly in slot 2
    [130,10],   ## segment 2 consumes mostly in slot 1
    [50,50]
])
lambda0 = np.array([6,1])  ## no purchase / static competitor
beta = 0.5                 ## rationality (feel free to change)
tmin, tmax = 0.0, 40.0     ## feasible price range for t

## revenue of a single price t
def revenue_of_t(t):
    lambda1 = np.array([t, 2])  ## contract 1 cheap in slot 2
    lambda2 = np.array([4, t])  ## contract 2 cheap in slot 1
    lambdas = [lambda0, lambda1, lambda2]
    total = 0.0
    for s in range(S):
        cost = np.array([np.inner(Es[s], lambdas[n]) for n in range(N+1)], dtype=float)
        exp_terms = np.exp(-beta * cost)
        probs = exp_terms / np.sum(exp_terms)
        total += np.sum(cost[1:] * probs[1:])  ## expected paid cost over purchasing options
    return total

## define energy U(x) = -revenue(t) with simple box projection
def U(x):
    t = float(np.clip(x[0], tmin, tmax))
    return -revenue_of_t(t)

## gradient dU via sympy if available, else finite-difference (kept minimal
import sympy as sp

_t = sp.Symbol('t', real=True)

Es_sym = sp.Matrix(Es)            #sympy version of Es
lambda0_sym = sp.Matrix(lambda0)  #sympy version of lambda0

_total = 0
for s in range(S):
    #contracts as sympy vectors
    lam1 = sp.Matrix([_t, 2.0])
    lam2 = sp.Matrix([4.0, _t])

    #row of Es as a 1x2 sympy matrix
    Es_row = Es_sym[s, :]

    #inner products are now pure sympy
    c0 = (Es_row * lambda0_sym)[0]
    c1 = (Es_row * lam1)[0]
    c2 = (Es_row * lam2)[0]

    exps = [sp.exp(-beta*c0), sp.exp(-beta*c1), sp.exp(-beta*c2)]
    Z = sum(exps)
    p0, p1, p2 = [e/Z for e in exps]
    _total += c1*p1 + c2*p2

_Uexpr  = -sp.simplify(_total)
_dUexpr = sp.diff(_Uexpr, _t)

_Ufun  = sp.lambdify(_t, _Uexpr,  modules='numpy')
_dUfun = sp.lambdify(_t, _dUexpr, modules='numpy')

def dU(x):
    t = float(np.clip(x[0], tmin, tmax))
    return np.array([float(_dUfun(t))], dtype=float)

## initial distribution (same style as your Rastrigin script)
def initial():
    return np.array([np.random.normal(20.0, 5.0)], dtype=float)

# --- Compute iterates according to algorithm (HRLA) ---
title = "HRLA_basic"
algorithm = GO.HRLA(d=1, M=1, N=1, K=5, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=False)

# --- Compute iterates according to ULA (baseline) ---
###samples_filename_ULA = algorithm.generate_samples(As=[1,2,3,4], sim_annealing=True)

# --- Plot empirical probabilities (same flow as yours) ---
postprocessor = PostProcessor(samples_filename_HRLA)
postprocessor.plot_empirical_probabilities(dpi=1, layout="11", tols=[1,2,3,4], running=False)

# --- Compute table of averages and standard deviations (same API) ---
postprocessor.compute_tables([5, 14], 1, "mean")
postprocessor.compute_tables([5, 14], 1, "std")

# --- Comparator HRLA vs ULA (same as your Rastrigin layout) ---
##comparator.plot_empirical_probabilities_per_d(dpi=1, tols=[3,4], running=True)
