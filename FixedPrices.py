import numpy as np
import sympy as sp
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor
import matplotlib.pyplot as plt
import os

# ============================================================
# EXTENSION (fixed fees introduced):
# - REAL weights
# - Multiple competitors, each with {flat, ToU} + annual fixed fee
# - Reservation bill = min competitor total bill (var + fixed)
# - Disutilities: DU_outside = 0, DU_n = bill_n - R_s
# - Retailer contracts have fixed fees (f1, f2)
# ============================================================

N_contracts = 2
S = 10
D = 2

beta = 0.02
tmin, tmax = 0.0, 20

Es_used = np.array([
    [3081.18, 267.45],
    [4727.04, 646.87],
    [2286.51, 499.64],
    [7308.55, 599.22],
    [1779.13, 344.83],
    [3740.28, 542.46],
    [2724.48, 824.06],
    [3383.92, 388.33],
    [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

w_data = np.array([
    0.125, 0.057, 0.068, 0.044, 0.161,
    0.116, 0.045, 0.171, 0.112, 0.101
], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# Competitors: now include annual fixed fees
# ============================================================

tou_A  = np.array([0.37, 0.28], dtype=float)
flat_A = np.array([0.345, 0.345], dtype=float)
F_A    = 160.0

tou_B  = np.array([0.35, 0.30], dtype=float)
flat_B = np.array([0.330, 0.330], dtype=float)
F_B    = 152.0

competitors = [flat_A, tou_A, flat_B, tou_B]
fixed_fees_comp_list = [F_A, F_A, F_B, F_B]

# Reservation bill per segment: min competitor TOTAL bill
competitors_np = np.array(competitors, dtype=float)  # (C,2)
Es_np = np.array(Es_used, dtype=float)               # (S,2)

comp_var_bills = Es_np @ competitors_np.T            # (S,C)
comp_fixed = np.array(fixed_fees_comp_list, dtype=float)  # (C,)
comp_bills = comp_var_bills + comp_fixed[None, :]    # (S,C)
R_data = comp_bills.min(axis=1)                      # (S,)

# ============================================================
# Retailer decision variables (with fixed fees)
# variables = (f1, p, f2, l2_day, l2_night)
# ============================================================

f1, p = sp.symbols("f1 p", real=True)
f2, l2_day, l2_night = sp.symbols("f2 l2_day l2_night", real=True)
variables = (f1, p, f2, l2_day, l2_night)

lambda1 = sp.Matrix([p, p])
lambda2 = sp.Matrix([l2_day, l2_night])
lambdas = [lambda1, lambda2]

fixed_fees = [f1, f2]

def symbolic_profit(beta_val: float, w: np.ndarray) -> sp.Expr:
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es_used[s])
        R_s = sp.Float(R_data[s])

        bills = [
            fixed_fees[n] + E_s.dot(lambdas[n])
            for n in range(N_contracts)
        ]

        DU_outside = sp.Integer(0)
        DUs = [DU_outside] + [b - R_s for b in bills]

        exp_terms = [sp.exp(-beta_val * u) for u in DUs]
        denom = sum(exp_terms)
        probs = [e / denom for e in exp_terms]

        total += w[s] * sum(bills[i] * probs[i + 1] for i in range(N_contracts))

    return total

profit_sym = symbolic_profit(beta, w_data)
U_sym = -profit_sym
gradU_sym = [sp.diff(U_sym, v) for v in variables]

U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

def U_x(x):
    return float(U_func(*x))

def dU_x(x):
    return np.array(gradU_func(*x), dtype=float).flatten()

# ============================================================
# Scaling (includes fixed-fee scale)
# ============================================================

comp_mean = np.mean(np.array(competitors, dtype=float), axis=0)  # (2,)
flat_scale = float(np.mean(comp_mean))
fixed_fee_scale = float(np.mean(fixed_fees_comp_list))  # average annual competitor fee

SCALE = np.array([
    fixed_fee_scale,   # f1
    flat_scale,        # p
    fixed_fee_scale,   # f2
    comp_mean[0],      # l2_day
    comp_mean[1],      # l2_night
], dtype=float)

def to_x(z):
    return SCALE * np.asarray(z, dtype=float)

def U_z(z):
    return U_x(to_x(z))

def dU_z(z):
    return SCALE * dU_x(to_x(z))

# ============================================================
# Run
# ============================================================

title = "FixedPrices"
d = len(variables)

def initial():
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(loc=0.0, scale=0.05, size=d)
    return z0

if __name__ == "__main__":
    print("Starting Optimization (N=2; 1 competitor; bill-based logit; UNIFORM weights; scaled variables)")

    algorithm = GO.HRLA(
        d=d,
        M=10,
        N=1,          # number of parallel trajectories (keep 1 unless you want more)
        K=2000,
        h=0.0001,
        title=title,
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[5],
        #As=[5, 50, 100, 500, 1000, 3000, 5000],
        sim_annealing=False,
    )

    print(f"Optimization finished. Samples saved to: {samples_filename}")
    print("NOTE: saved samples are z-vectors. Convert to real tariffs via x = to_x(z).")

    # --- Minimal postprocessing (tables + best trajectories) ---
    postprocessor = PostProcessor(samples_filename)

    #measured = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    measured = [1, 3, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)
