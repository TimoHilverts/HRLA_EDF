import numpy as np
import time
from scipy.optimize import minimize

# ============================================================
# MODEL:
# Retailer contracts (2):
#   (R1) flat: p
#   (R2) ToU : (l2_day, l2_night)
#
# Competitor contracts (2) -> outside options:
#   (C1) ToU : (0.355, 0.250) + Fixed Fee 196
#   (C2) flat: 0.342 + Fixed Fee 196
# ============================================================

S = 10
beta = 5
COMP_FIXED_FEE = 196.0

# --------------------------
# Consumption (kWh)
# --------------------------
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

# --------------------------
# Competitors
# --------------------------
comp_tou = np.array([0.355, 0.250], dtype=float)
comp_flat = 0.342

# Uniform weights
w_uniform = np.ones(S, dtype=float) / S

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit + analytic gradient (x-space)
# x = [p, l2_day, l2_night]
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    p, l2_day, l2_night = map(float, x)

    profit = 0.0
    grad_profit = np.zeros(3, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]
        T = E_day + E_night

        # Bills
        b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night + COMP_FIXED_FEE
        b_c2 = comp_flat * T + COMP_FIXED_FEE
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night
        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)

        # Expected retailer revenue
        r = b_r1 * P[2] + b_r2 * P[3]
        profit += w_s * r

        # Derivatives
        db_dp = np.array([0.0, 0.0, T, 0.0], dtype=float)
        db_ld = np.array([0.0, 0.0, 0.0, E_day], dtype=float)
        db_ln = np.array([0.0, 0.0, 0.0, E_night], dtype=float)

        def dr_direction(db_dtheta: np.ndarray) -> float:
            du = -beta * db_dtheta
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return (
                db_dtheta[2] * P[2] + b_r1 * dP[2]
                + db_dtheta[3] * P[3] + b_r2 * dP[3]
            )

        grad_profit[0] += w_s * dr_direction(db_dp)
        grad_profit[1] += w_s * dr_direction(db_ld)
        grad_profit[2] += w_s * dr_direction(db_ln)

    return profit, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)  # minimize negative profit

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# ============================================================
# Scaling wrapper (z-space), same idea as HRLA
# x = SCALE * z
# ============================================================
SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def to_z(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / SCALE

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))

# ============================================================
# Gradient checker
# ============================================================
def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g

# ============================================================
# Bounds in z-space
# These match the broad DE-style search region
# ============================================================
bounds_z = [
    (0.5, 2.0),   # p / comp_flat
    (0.5, 2.0),   # l2_day / comp_tou_day
    (0.5, 3.0),   # l2_night / comp_tou_night
]

# ============================================================
# Random initializer in z-space
# ============================================================
def random_initial_z(rng: np.random.Generator) -> np.ndarray:
    return np.array([
        rng.uniform(bounds_z[0][0], bounds_z[0][1]),
        rng.uniform(bounds_z[1][0], bounds_z[1][1]),
        rng.uniform(bounds_z[2][0], bounds_z[2][1]),
    ], dtype=float)

# ============================================================
# Single L-BFGS-B run
# ============================================================
def run_lbfgsb_single(z0: np.ndarray):
    result = minimize(
        fun=U_z,
        x0=np.asarray(z0, dtype=float),
        method="L-BFGS-B",
        jac=dU_z,
        bounds=bounds_z,
        options={
            "maxiter": 5000,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )

    z_best = result.x
    x_best = to_x(z_best)
    profit_best, _ = profit_and_grad(x_best)

    return result, z_best, x_best, profit_best

# ============================================================
# Multi-start L-BFGS-B
# ============================================================
def run_multistart_lbfgsb(n_starts: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    all_profits = []
    all_z = []
    all_x = []
    all_fun = []
    all_success = []
    all_nit = []
    all_nfev = []
    all_messages = []

    best_profit = -np.inf
    best_result = None
    best_z = None
    best_x = None

    for i in range(n_starts):
        z0 = random_initial_z(rng)

        result, z_best, x_best, profit_best = run_lbfgsb_single(z0)

        all_profits.append(profit_best)
        all_z.append(z_best)
        all_x.append(x_best)
        all_fun.append(result.fun)
        all_success.append(result.success)
        all_nit.append(result.nit)
        all_nfev.append(result.nfev)
        all_messages.append(result.message)

        if profit_best > best_profit:
            best_profit = profit_best
            best_result = result
            best_z = z_best
            best_x = x_best

    end_time = time.perf_counter()
    runtime = end_time - start_time

    summary = {
        "runtime_seconds": runtime,
        "profits": np.array(all_profits, dtype=float),
        "z_solutions": np.array(all_z, dtype=float),
        "x_solutions": np.array(all_x, dtype=float),
        "fun_values": np.array(all_fun, dtype=float),
        "success_flags": np.array(all_success, dtype=bool),
        "iterations": np.array(all_nit, dtype=int),
        "function_evals": np.array(all_nfev, dtype=int),
        "messages": all_messages,
        "best_profit": float(best_profit),
        "best_z": np.array(best_z, dtype=float),
        "best_x": np.array(best_x, dtype=float),
        "best_result": best_result,
    }

    return summary

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Gradient check in x-space
    x_test = to_x(np.ones(3, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    print("\nRunning multi-start L-BFGS-B...\n")

    summary = run_multistart_lbfgsb(n_starts=100, seed=0)

    profits = summary["profits"]
    best_result = summary["best_result"]

    print("\n========== MULTI-START L-BFGS-B RESULT ==========\n")
    print("Number of starts        :", len(profits))
    print("Successful runs         :", int(np.sum(summary["success_flags"])))
    print("Mean profit (€)         :", float(np.mean(profits)))
    print("Std profit (€)          :", float(np.std(profits)))
    print("Best profit (€)         :", summary["best_profit"])
    print("Worst profit (€)        :", float(np.min(profits)))

    print("\nBest z (scaled)         :", summary["best_z"])
    print("Best x (tariffs)        :", summary["best_x"])
    print("  Flat price p          :", summary["best_x"][0])
    print("  ToU day price         :", summary["best_x"][1])
    print("  ToU night price       :", summary["best_x"][2])

    print("\nTotal runtime (seconds) :", summary["runtime_seconds"])
    print("Mean iterations/run     :", float(np.mean(summary["iterations"])))
    print("Mean function evals/run :", float(np.mean(summary["function_evals"])))
    print("Best-run iterations     :", best_result.nit)
    print("Best-run function evals :", best_result.nfev)
    print("Best-run message        :", best_result.message)

    print("\n=================================================\n")