import numpy as np
import time
import cma  # pip install cma
from scipy.optimize import Bounds

# ============================================================
# DATA & PARAMETERS (Full Model)
# ============================================================
S = 10
beta = 0.02
shift = 0.12

C_DAY, C_NIGHT, C_GREEN, C_SERVE = 0.29, 0.21, 0.010, 110.0

Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

def shifting_cons(E, s):
    return np.array([(1.0 - s) * E[0], E[1] + s * E[0]], dtype=float)

Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)
w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161, 0.116, 0.045, 0.171, 0.112, 0.101], dtype=float)
w_data /= np.sum(w_data)

tou_A, flat_A, F_A = np.array([0.355, 0.250]), 0.342, 196.0
competitor_np = np.array([[flat_A, flat_A], [tou_A[0], tou_A[1]]])
fixed_fees_comp = np.array([F_A, F_A])
R_data = ((Es_used @ competitor_np.T) + fixed_fees_comp[None, :]).min(axis=1)
B_reg_data = F_A + (Es_used @ np.mean(competitor_np, axis=0))
g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00, 0.02, 0.02, 0.00, 0.00, 0.04])
delta = np.array([0.0, 1.0, 0.0, 1.0])

# ============================================================
# COST & PROFIT FUNCTIONS
# ============================================================
def cost_per_contract(E_day, E_night):
    T = E_day + E_night
    total_d, total_n = np.sum(Es_original[:, 0]), np.sum(Es_original[:, 1])
    c_flat = (total_d * C_DAY + total_n * C_NIGHT) / (total_d + total_n)
    c_tou_base = C_SERVE + (C_DAY * E_day + C_NIGHT * E_night)
    c_flat_base = C_SERVE + (c_flat * T)
    return np.array([c_flat_base, c_flat_base + (C_GREEN * T), c_tou_base, c_tou_base + (C_GREEN * T)])

def profit(x):
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    total_p = 0.0
    for s in range(S):
        E_day, E_night = Es_used[s]
        T, w_s, R_s, g_s, B_s = E_day + E_night, w_data[s], R_data[s], g_levels[s], B_reg_data[s]
        bills = np.array([f1 + p1*T, f2 + p2*T, f3 + l3d*E_day + l3n*E_night, f4 + l4d*E_day + l4n*E_night])
        u = -beta * np.concatenate(([0.0], (bills - delta * g_s * B_s) - R_s))
        P = np.exp(u - np.max(u)); P /= P.sum()
        total_p += w_s * np.dot(bills - cost_per_contract(E_day, E_night), P[1:])
    return total_p

# Scaling
SCALE = np.array([F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]])
def U_z(z): return -profit(SCALE * z)

# ============================================================
# CMA-ES RUNNER
# ============================================================
def run_cma_es(seed: int = 0, sigma0: float = 0.25):
    # CMA-ES Setup
    # x0 is the initial guess (z=1 means matching Competitor A)
    x0 = np.ones(10)
    
    # Bounds in z-space [0.5, 3.0]
    lb, ub = 0.5, 3.0
    opts = {
        'popsize': 10,
        'bounds': [lb, ub],
        'seed': seed,
        'verb_log': 0,
        'tolfun': 1e-9,
        'maxiter': 10000
    }

    start = time.perf_counter()
    
    # es.result returns: (xbest, fbest, evals_best, evaluations, iterations, xfavorite, stds)
    res = cma.fmin(U_z, x0, sigma0, opts)
    
    end = time.perf_counter()
    
    x_best_scaled = SCALE * res[0]
    profit_best = -res[1]
    runtime = end - start
    
    return res, x_best_scaled, profit_best, runtime

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    N_RUNS = 10
    SIGMA0 = 0.25 # Initial search volume/step-size

    profits, runtimes = [], []
    best_overall_profit = -np.inf
    best_overall_x = None

    print(f"Running CMA-ES ({N_RUNS} runs, sigma0={SIGMA0})...\n")
    total_start = time.perf_counter()

    for i in range(N_RUNS):
        res, x_best, p_best, run_t = run_cma_es(seed=i, sigma0=SIGMA0)
        
        profits.append(p_best)
        runtimes.append(run_t)

        if p_best > best_overall_profit:
            best_overall_profit = p_best
            best_overall_x = x_best
        
        print(f"  Run {i+1:2d}: Profit = {p_best:.8f} | Time = {run_t:.8f} s")

    total_end = time.perf_counter()

    # Final Summary Statistics
    labels = ["f1", "p1", "f2", "p2", "f3", "l3_day", "l3_night", "f4", "l4_day", "l4_night"]

    print("\n" + "="*65)
    print(f"STATISTICAL SUMMARY (Over {N_RUNS} CMA-ES Runs)")
    print("="*65)
    print(f"Mean Profit         : {np.mean(profits):.8f} €")
    print(f"Std. Dev. Profit    : {np.std(profits):.8f} €")
    print(f"Best Profit Found   : {best_overall_profit:.8f} €")
    print("-" * 65)
    print(f"Total Computation   : {total_end - total_start:.8f} s")
    print(f"Mean Runtime / Run  : {np.mean(runtimes):.8f} s")
    print(f"Std. Runtime / Run  : {np.std(runtimes):.8f} s")
    print("="*65)

    print("\n========== OPTIMAL CONTRACT PARAMETERS (BEST CMA RUN) ==========\n")
    for lbl, val in zip(labels, best_overall_x):
        unit = "€/yr" if lbl.startswith("f") else "€/kWh"
        print(f"  {lbl:10s} : {val:.8f} {unit}")