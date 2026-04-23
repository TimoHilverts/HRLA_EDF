import numpy as np
import time
from scipy.optimize import differential_evolution

# ============================================================
# MODEL
# ============================================================

S = 10
beta = 0.03
COMP_FIXED_FEE = 196.0

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

comp_tou = np.array([0.355, 0.250])
comp_flat = 0.342

w_uniform = np.ones(S) / S

# ============================================================
# Stable softmax
# ============================================================

def softmax_stable(u):
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit function
# ============================================================

def profit(x):

    p, l2_day, l2_night = x

    total_profit = 0.0

    for s in range(S):

        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]

        T = E_day + E_night

        # Bills
        b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night + COMP_FIXED_FEE
        b_c2 = comp_flat * T + COMP_FIXED_FEE
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night

        b = np.array([b_c1, b_c2, b_r1, b_r2])

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)

        # Revenue
        r = b_r1 * P[2] + b_r2 * P[3]

        total_profit += w_s * r

    return total_profit

# ============================================================
# Objective
# ============================================================

def objective(x):
    return -profit(x)

# ============================================================
# Scaling (same as HRLA)
# ============================================================

SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]])

def to_x(z):
    return SCALE * np.asarray(z)

def objective_z(z):
    return objective(to_x(z))

# ============================================================
# Bounds in scaled space
# ============================================================

bounds = [
    (0.5, 2.0),
    (0.5, 2.0),
    (0.5, 2.5),
]

# ============================================================
# Run multiple experiments
# ============================================================

if __name__ == "__main__":

    N_RUNS = 20

    revenues = []
    runtimes = []

    best_overall = -np.inf
    best_x = None

    print("\nRunning Differential Evolution experiments...\n")

    for i in range(N_RUNS):

        start = time.perf_counter()

        result = differential_evolution(
            objective_z,
            bounds=bounds,
            maxiter=2000,
            popsize=25,
            tol=1e-7,
            polish=True,
            seed=i
        )

        end = time.perf_counter()

        x_best = to_x(result.x)
        revenue = profit(x_best)

        revenues.append(revenue)
        runtimes.append(end - start)

        if revenue > best_overall:
            best_overall = revenue
            best_x = x_best

        print(f"Run {i+1:2d}: revenue = {revenue:.4f}")

    revenues = np.array(revenues)
    runtimes = np.array(runtimes)

    # ========================================================
    # Statistics
    # ========================================================

    mean_rev = np.mean(revenues)
    std_rev = np.std(revenues)

    mean_time = np.mean(runtimes)
    std_time = np.std(runtimes)

    print("\n========== DIFFERENTIAL EVOLUTION SUMMARY ==========\n")

    print("Best tariffs found:")
    print("Flat price p      :", best_x[0])
    print("ToU day price     :", best_x[1])
    print("ToU night price   :", best_x[2])

    print("\nBest revenue      :", best_overall)

    print("\nMean revenue      :", mean_rev)
    print("Std revenue       :", std_rev)

    print("\nMean runtime (s)  :", mean_time)
    print("Std runtime (s)   :", std_time)

    print("\n====================================================\n")