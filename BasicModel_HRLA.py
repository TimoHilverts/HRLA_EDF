import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ---------------------------------------------------
# Model parameters
# ---------------------------------------------------
N = 2  
S = 3  
D = 2  

Es = np.array([
    [10, 120],
    [130, 10],
    [50, 50]
])
outside = np.array([6, 1])

tmin, tmax = 0.0, 40.0

# ---------------------------------------------------
# SYMBOLIC SETUP (generic for variable beta)
# ---------------------------------------------------
t = sp.Symbol("t", real=True)

lambda0 = sp.Matrix(outside)
lambda1 = sp.Matrix([t, 2])
lambda2 = sp.Matrix([4, t])
lambdas = [lambda0, lambda1, lambda2]

def make_symbolic_revenue(beta):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        costs = [E_s.dot(lam) for lam in lambdas]
        max_c = sp.Max(*[-beta * c for c in costs])
        shifted_exp = [sp.exp(-beta * c - max_c) for c in costs]
        denom = sum(shifted_exp)
        probs = [e/denom for e in shifted_exp]
        total += sum(costs[i] * probs[i] for i in range(1, N+1))
    return total

# ---------------------------------------------------
# Construct HRLA objective U and dU for a given beta
# ---------------------------------------------------
def build_functions(beta):
    revenue_sym = make_symbolic_revenue(beta)
    U_sym = -revenue_sym
    dU_sym = sp.diff(U_sym, t)

    U_func = sp.lambdify(t, U_sym, "numpy")
    dU_func = sp.lambdify(t, dU_sym, "numpy")

    def U(x):
        tv = float(np.clip(x[0], tmin, tmax))
        return float(U_func(tv))

    def dU(x):
        tv = float(np.clip(x[0], tmin, tmax))
        return np.array([float(dU_func(tv))], dtype=float)

    return U, dU

# ---------------------------------------------------
# HRLA CONFIG
# ---------------------------------------------------
d = 1
def initial():
    return np.random.multivariate_normal(np.zeros(d)+3, 10*np.eye(d))

As = [100]
Ks = [1500]

# ---------------------------------------------------
# Sweep over betas and store resulting max revenues
# ---------------------------------------------------
beta_vals = np.logspace(-2, 2, 20)
revenues = []

for beta in beta_vals:
    print(f"Running HRLA for beta = {beta:.4f}")

    U, dU = build_functions(beta)

    algorithm = GO.HRLA(
        d=d, M=1, N=1, K=1500, h=0.01,
        title=f"HRLA_beta_{beta}",
        U=U, dU=dU, initial=initial
    )

    samplefile = algorithm.generate_samples(As=As, sim_annealing=False)

    post = PostProcessor(samplefile)
    bests = post.get_best(measured=[1500], dpi=1)

    t_opt = bests[0][0]   # HRLA optimal price (dimension 1)
    R_opt = -U([t_opt])   # revenue = -U

    revenues.append(R_opt)

# ---------------------------------------------------
# Plot Beta vs Max Revenue
# ---------------------------------------------------
plt.figure(figsize=(8,5))
plt.plot(beta_vals, revenues, marker='o', markersize=5)
plt.xscale('log')
plt.xlabel("β (consumer rationality)")
plt.ylabel("Maximum Revenue at HRLA-optimal t")
plt.title("β–Revenue Curve (using HRLA optimal prices)")
plt.grid(True, which='both')
plt.show()
