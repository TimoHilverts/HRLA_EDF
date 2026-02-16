import numpy as np
import sympy as sp
from sympy import Piecewise, Eq  # kept for consistency with your project imports
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor, Comparator  # Comparator kept for consistency
import matplotlib.pyplot as plt  # kept for consistency (not used here)
import os  # kept for consistency

# ============================================================
# BASELINE MODEL:
# - N=2 contracts:
#   (1) flat (same day/night)
#   (2) ToU (day/night)
# - Outside option: ONE competitor (variable fees only)
# - Choice model: exp(-beta * BILL) directly (no disutilities)
# - Segment weights: UNIFORM (1/S each)
# ============================================================

N_contracts = 2
S = 10
D = 2

beta = 0.02
tmin, tmax = 0.0, 20  # optional bounds if you later want clipping

# Consumption in kWh
Es_used = np.array([
    [3081.18, 267.45],  # segment 0
    [4727.04, 646.87],  # segment 1
    [2286.51, 499.64],  # segment 2
    [7308.55, 599.22],  # segment 3
    [1779.13, 344.83],  # segment 4
    [3740.28, 542.46],  # segment 5
    [2724.48, 824.06],  # segment 6
    [3383.92, 388.33],  # segment 7
    [1762.53, 224.38],  # segment 8
    [4896.07, 401.88],  # segment 9
], dtype=float)

# One competitor (outside option): variable fees only (day, night)
competitor = np.array([0.37, 0.28], dtype=float)
competitor_sym = sp.Matrix(competitor)

# ============================================================
# UNIFORM segment weights (baseline)
# ============================================================
w_uniform = np.ones(S, dtype=float) / S  # w_s = 1/10

# ============================================================
# Decision variables (N=2, d=3)
# Contract 1: flat price p (same day/night)
# Contract 2: ToU prices (l2_day, l2_night)
# ============================================================

p = sp.symbols("p", real=True)
l2_day, l2_night = sp.symbols("l2_day l2_night", real=True)

variables = (p, l2_day, l2_night)

lambda1 = sp.Matrix([p, p])                 # flat
lambda2 = sp.Matrix([l2_day, l2_night])     # ToU
lambdas = [lambda1, lambda2]

def outside_bill(E_s: sp.Matrix) -> sp.Expr:
    """Outside option bill under the single competitor."""
    return E_s.dot(competitor_sym)

def symbolic_profit(beta_val: float, w: np.ndarray) -> sp.Expr:
    """
    Expected retailer revenue under bill-based logit:
        P(choice=j) ∝ exp(-beta * bill_j)
    Alternatives include the outside option (competitor).
    Outside option contributes 0 to retailer revenue.
    """
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es_used[s])

        # bills for retailer contracts
        bills = [E_s.dot(lam) for lam in lambdas]  # length 2

        # outside bill
        b_out = outside_bill(E_s)

        # bill-based logit over {outside, contract 1, contract 2}
        exp_terms = [sp.exp(-beta_val * b_out)] + [sp.exp(-beta_val * b) for b in bills]
        denom = sum(exp_terms)
        probs = [e / denom for e in exp_terms]  # probs[0]=outside, probs[1]=c1, probs[2]=c2

        # expected revenue (outside yields 0)
        total += w[s] * (bills[0] * probs[1] + bills[1] * probs[2])

    return total

# Use UNIFORM weights here (baseline)
profit_sym = symbolic_profit(beta, w_uniform)
U_sym = -profit_sym
gradU_sym = [sp.diff(U_sym, v) for v in variables]

# Numerical evaluation
U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

def U_x(x):
    # Optional clipping:
    # x = np.clip(np.asarray(x, dtype=float), tmin, tmax)
    return float(U_func(*x))

def dU_x(x):
    # Optional clipping:
    # x = np.clip(np.asarray(x, dtype=float), tmin, tmax)
    return np.array(gradU_func(*x), dtype=float).flatten()

# ============================================================
# SCALING WRAPPER (z-space), consistent with competitor tariffs
# ============================================================

# Scale flat price with the competitor mean; scale ToU components with competitor day/night
lambda_scale = competitor.copy()                 # [day, night]
flat_scale = float(np.mean(lambda_scale))        # scalar scale for p

SCALE = np.array([
    flat_scale,           # p
    lambda_scale[0],      # l2_day
    lambda_scale[1],      # l2_night
], dtype=float)

def to_x(z):
    """Map scaled variables z -> original variables x."""
    return SCALE * np.asarray(z, dtype=float)

def to_z(x):
    """Map original variables x -> scaled variables z."""
    return np.asarray(x, dtype=float) / SCALE

def U_z(z):
    return U_x(to_x(z))

def dU_z(z):
    # Chain rule: d/dz U(SCALE*z) = SCALE * d/dx U(x) evaluated at x=SCALE*z
    return SCALE * dU_x(to_x(z))

def segment_choice_probs_x(x):
    """
    Segment-level choice probabilities evaluated at x.
    Returns P of shape (S, 3) with columns:
        [Outside, Contract 1 (flat), Contract 2 (ToU)].
    """
    x = np.asarray(x, dtype=float).flatten()
    if x.size != 3:
        raise ValueError(f"Expected x of length 3, got {x.size}.")

    p_val, l2d_val, l2n_val = x

    # Bills for each segment under retailer contracts
    # Contract 1: flat p
    bill_c1 = Es_used[:, 0] * p_val + Es_used[:, 1] * p_val
    # Contract 2: ToU (l2_day, l2_night)
    bill_c2 = Es_used[:, 0] * l2d_val + Es_used[:, 1] * l2n_val

    # Outside option bill
    bill_out = Es_used[:, 0] * competitor[0] + Es_used[:, 1] * competitor[1]

    # Logit over [outside, c1, c2] with exp(-beta * bill)
    e_out = np.exp(-beta * bill_out)
    e_c1  = np.exp(-beta * bill_c1)
    e_c2  = np.exp(-beta * bill_c2)
    denom = e_out + e_c1 + e_c2

    P = np.column_stack([e_out / denom, e_c1 / denom, e_c2 / denom])
    return P


def segment_choice_probs(z_or_x):
    """
    Convenience wrapper for postprocessing:
    - If input looks like z (scaled, around ~1), it converts z -> x.
    - If you prefer explicitness, call segment_choice_probs_x(x) directly.
    """
    v = np.asarray(z_or_x, dtype=float).flatten()
    if v.size != 3:
        raise ValueError(f"Expected length 3, got {v.size}.")
    # Heuristic: if values are around 1, assume z-space; otherwise x-space.
    if np.all((v > 0.05) & (v < 5.0)):
        # likely z
        x = to_x(v)
    else:
        x = v
    return segment_choice_probs_x(x)


# ============================================================
# COMPUTING + POSTPROCESSING (minimal)
# ============================================================

title = "BasicModel_N2_1competitor_billlogit_uniformweights_scaled"
d = 3

def initial():
    """
    Initial point in z-space.
    z ≈ [1,1,1] corresponds to x ≈ SCALE (competitor-like tariffs).
    """
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(loc=0.0, scale=0.05, size=d)
    return z0

if __name__ == "__main__":
    print("Starting Optimization (N=2; 1 competitor; bill-based logit; UNIFORM weights; scaled variables)")

    algorithm = GO.HRLA(
        d=d,
        M=1,
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
