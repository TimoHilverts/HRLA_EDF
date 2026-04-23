import numpy as np
import time
from scipy.optimize import basinhopping, Bounds

# ============================================================
# DATA & PARAMETERS (Full Model)
# ============================================================
S = 10
beta = 0.02  # Choice sensitivity
shift = 0.12 # 12% consumption shifting

# Explicit Cost Structure (Euros)
C_DAY   = 0.29
C_NIGHT = 0.21
C_GREEN = 0.010
C_SERVE = 110.0

# Consumption Data
Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

def shifting_cons(E, s):
    return np.array([(1.0 - s) * E[0], E[1] + s * E[0]], dtype=float)

Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161,
                    0.116, 0.045, 0.171, 0.112, 0.101], dtype=float)
w_data /= np.sum(w_data)

# Competitor A Reference
tou_A, flat_A, F_A = np.array([0.355, 0.250]), 0.342, 196.0
competitor_np = np.array([[flat_A, flat_A], [tou_A[0], tou_A[1]]])
fixed_fees_comp = np.array([F_A, F_A])

# Reservation Bill R_s and Benchmark B_reg
comp_bills = (Es_used @ competitor_np.T) + fixed_fees_comp[None, :]
R_data = comp_bills.min(axis=1)
lambda_reg = np.mean(competitor_np, axis=0)
B_reg_data = F_A + (Es_used @ lambda_reg)

# Green preferences
g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00, 0.02, 0.02, 0.00, 0.00, 0.04])
delta = np.array([0.0, 1.0, 0.0, 1.0])

# ============================================================
# CORE FUNCTIONS (Profit & Analytic Gradient)
# ============================================================
def cost_per_contract(E_day, E_night):
    T = E_day + E_night
    # Weighted average cost for flat logic
    total_d, total_n = np.sum(Es_original[:, 0]), np.sum(Es_original[:, 1])
    c_flat = (total_d * C_DAY + total_n * C_NIGHT) / (total_d + total_n)
    
    c_tou_base = C_SERVE + (C_DAY * E_day + C_NIGHT * E_night)
    c_flat_base = C_SERVE + (c_flat * T)
    return np.array([c_flat_base, c_flat_base + (C_GREEN * T), 
                     c_tou_base, c_tou_base + (C_GREEN * T)])

def profit_and_grad(x):
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    total_profit = 0.0
    grad_profit = np.zeros(10)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T, w_s, R_s, g_s, B_reg_s = E_day + E_night, w_data[s], R_data[s], g_levels[s], B_reg_data[s]
        
        bills = np.array([f1 + p1*T, f2 + p2*T, f3 + l3d*E_day + l3n*E_night, f4 + l4d*E_day + l4n*E_night])
        costs = cost_per_contract(E_day, E_night)
        margins = bills - costs

        u = -beta * np.concatenate(([0.0], (bills - delta * g_s * B_reg_s) - R_s))
        P = np.exp(u - np.max(u)); P /= P.sum()

        total_profit += w_s * np.dot(margins, P[1:])

        def dp_dir(dbills):
            db = np.array(dbills); du = -beta * np.concatenate(([0.0], db))
            dP = P * (du - np.dot(P, du))
            return np.dot(db, P[1:]) + np.dot(margins, dP[1:])

        grad_profit += w_s * np.array([
            dp_dir([1,0,0,0]), dp_dir([T,0,0,0]), dp_dir([0,1,0,0]), dp_dir([0,T,0,0]),
            dp_dir([0,0,1,0]), dp_dir([0,0,E_day,0]), dp_dir([0,0,E_night,0]),
            dp_dir([0,0,0,1]), dp_dir([0,0,0,E_day]), dp_dir([0,0,0,E_night])
        ])
    return total_profit, grad_profit

# ============================================================
# SCALING & OPTIMIZATION WRAPPERS
# ============================================================
SCALE = np.array([F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]])

def U_z(z): return -profit_and_grad(SCALE * z)[0]
def dU_z(z): return -SCALE * profit_and_grad(SCALE * z)[1]

class RandomDisplacementBounds:
    def __init__(self, xmin, xmax, stepsize=0.15):
        self.xmin, self.xmax, self.stepsize = xmin, xmax, stepsize
    def __call__(self, x):
        return np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, x.shape), self.xmin, self.xmax)

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    N_RUNS, NITER, STEPSIZE = 10, 200, 0.20
    profits, runtimes = [], []
    best_overall_profit, best_res = -np.inf, None

    lower, upper = np.ones(10) * 0.5, np.ones(10) * 3.0
    take_step = RandomDisplacementBounds(lower, upper, stepsize=STEPSIZE)

    print(f"Running Basin-Hopping ({N_RUNS} runs, {NITER} iterations each)...\n")
    total_start = time.perf_counter()

    for seed in range(N_RUNS):
        start = time.perf_counter()
        res = basinhopping(U_z, x0=np.ones(10), niter=NITER, take_step=take_step, 
                           minimizer_kwargs={"method": "L-BFGS-B", "jac": dU_z, "bounds": Bounds(lower, upper)},
                           seed=seed)
        end = time.perf_counter()
        
        current_profit = -res.fun
        profits.append(current_profit)
        runtimes.append(end - start)

        if current_profit > best_overall_profit:
            best_overall_profit, best_res = current_profit, res
        
        print(f"  Run {seed+1:2d}: Profit = {current_profit:.8f} | Time = {end-start:.8f} s")

    total_end = time.perf_counter()

    # Final Stats
    labels = ["f1", "p1", "f2", "p2", "f3", "l3_day", "l3_night", "f4", "l4_day", "l4_night"]
    x_best = SCALE * best_res.x

    print("\n" + "="*65)
    print(f"STATISTICAL SUMMARY (Over {N_RUNS} Basin-Hopping Runs)")
    print("="*65)
    print(f"Mean Profit         : {np.mean(profits):.8f} €")
    print(f"Std. Dev. Profit    : {np.std(profits):.8f} €")
    print(f"Best Profit Found   : {best_overall_profit:.8f} €")
    print("-" * 65)
    print(f"Total Computation   : {total_end - total_start:.8f} s")
    print(f"Mean Runtime / Run  : {np.mean(runtimes):.8f} s")
    print(f"Std. Runtime / Run  : {np.std(runtimes):.8f} s")
    print("="*65)

    print("\n========== OPTIMAL CONTRACT PARAMETERS (BEST RUN) ==========\n")
    for lbl, val in zip(labels, x_best):
        unit = "€/yr" if lbl.startswith("f") else "€/kWh"
        print(f"  {lbl:10s} : {val:.8f} {unit}")