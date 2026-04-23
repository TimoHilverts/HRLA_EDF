import numpy as np
import time
from scipy.optimize import differential_evolution

# ============================================================
# EXTENSION: Single Competitor + Consumption Shifting
# - Retailer contracts: Flat (f1, p) and ToU (f2, l2_day, l2_night)
# - Static competitive environment: Competitor A only (with fixed fee)
# - Consumption shifting: 12% shift from day to night
# - Choice model: logit over {outside, R1, R2}
# - Optimization: Differential Evolution
# ============================================================

S = 10
beta = 0.03

# --------------------------
# Consumption (kWh): original
# --------------------------
Es_original = np.array([
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

def shifting_cons(E: np.ndarray, shift: float) -> np.ndarray:
    E_day, E_night = E
    E_day_new = (1.0 - shift) * E_day
    E_night_new = E_night + shift * E_day
    return np.array([E_day_new, E_night_new], dtype=float)

shift = 0.12
Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

# --------------------------
# Segment weights: empirical
# --------------------------
w_data= np.array([
    0.125, 0.057, 0.068, 0.044, 0.161,
    0.116, 0.045, 0.171, 0.112, 0.101
], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# Single competitor
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

# Competitor tariffs
competitor_np = np.array([
    [flat_A, flat_A],      # Flat
    [tou_A[0], tou_A[1]]   # ToU
], dtype=float)

fixed_fees_comp = np.array([F_A, F_A], dtype=float)

Es_np = np.array(Es_used, dtype=float)
comp_var_bills = Es_np @ competitor_np.T
comp_bills = comp_var_bills + fixed_fees_comp[None, :]

# Reservation bill: best competitor option
R_data = comp_bills.min(axis=1)

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit and gradient
# x = [f1, p, f2, l2_day, l2_night]
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    f1, p, f2, l2_day, l2_night = map(float, x)
    profit = 0.0
    grad_profit = np.zeros(5, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s = float(w_data[s])
        R_s = float(R_data[s])

        # Retailer bills
        b1 = f1 + p * T
        b2 = f2 + l2_day * E_day + l2_night * E_night

        # Logit over {outside, R1, R2}
        DU = np.array([0.0, b1 - R_s, b2 - R_s], dtype=float)
        u = -beta * DU
        P = softmax_stable(u)

        # Expected retailer revenue
        r = b1 * P[1] + b2 * P[2]
        profit += w_s * r

        def dr_direction(db1: float, db2: float) -> float:
            dDU = np.array([0.0, db1, db2], dtype=float)
            du = -beta * dDU
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return db1 * P[1] + b1 * dP[1] + db2 * P[2] + b2 * dP[2]

        grad_profit[0] += w_s * dr_direction(db1=1.0, db2=0.0)      # df1
        grad_profit[1] += w_s * dr_direction(db1=T,   db2=0.0)      # dp
        grad_profit[2] += w_s * dr_direction(db1=0.0, db2=1.0)      # df2
        grad_profit[3] += w_s * dr_direction(db1=0.0, db2=E_day)    # dl2_day
        grad_profit[4] += w_s * dr_direction(db1=0.0, db2=E_night)  # dl2_night

    return profit, grad_profit

# ============================================================
# Objective functions
# ============================================================
def profit_only(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return float(prof)

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# ============================================================
# Scaling
# ============================================================
SCALE = np.array([F_A, flat_A, F_A, tou_A[0], tou_A[1]], dtype=float)

def to_z(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / SCALE

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

# ============================================================
# Finite-difference gradient checker
# ============================================================
def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g

# ============================================================
# Differential Evolution bounds in scaled space
# ============================================================
bounds = [
    (0.0, 3.0),  # z_f1
    (0.0, 3.0),  # z_p
    (0.0, 3.0),  # z_f2
    (0.0, 3.0),  # z_l2_day
    (0.0, 3.0),  # z_l2_night
]

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    d = 5

    # ---- Gradient check ----
    x_test = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    # ---- Differential Evolution runs ----
    N_RUNS = 20

    revenues = []
    runtimes = []

    best_overall = -np.inf
    best_x = None
    best_z = None
    best_result = None

    print(f"\nRunning Differential Evolution (shift={shift}, beta={beta}, F_A={F_A})\n")

    for i in range(N_RUNS):
        start = time.perf_counter()

        result = differential_evolution(
            U_z,
            bounds=bounds,
            maxiter=2000,
            popsize=25,
            tol=1e-7,
            polish=True,
            seed=i
        )

        end = time.perf_counter()

        z_best = result.x
        x_best = to_x(z_best)
        revenue = profit_only(x_best)

        revenues.append(revenue)
        runtimes.append(end - start)

        if revenue > best_overall:
            best_overall = revenue
            best_x = x_best.copy()
            best_z = z_best.copy()
            best_result = result

        print(f"Run {i+1:2d}: revenue = {revenue:.6f}")

    revenues = np.array(revenues)
    runtimes = np.array(runtimes)

    mean_rev = np.mean(revenues)
    std_rev = np.std(revenues)
    best_rev = np.max(revenues)
    worst_rev = np.min(revenues)

    mean_time = np.mean(runtimes)
    std_time = np.std(runtimes)

    print("\n========== DIFFERENTIAL EVOLUTION SUMMARY ==========\n")

    print("Best scaled vector z:")
    print(best_z)

    print("\nBest tariffs found x = [f1, p, f2, l2_day, l2_night]:")
    print("Retailer flat fixed fee f1 :", best_x[0])
    print("Retailer flat usage price p :", best_x[1])
    print("Retailer ToU fixed fee f2   :", best_x[2])
    print("Retailer ToU day price      :", best_x[3])
    print("Retailer ToU night price    :", best_x[4])

    print("\nBest revenue      :", best_overall)
    print("Mean revenue      :", mean_rev)
    print("Std revenue       :", std_rev)
    print("Worst revenue     :", worst_rev)

    print("\nMean runtime (s)  :", mean_time)
    print("Std runtime (s)   :", std_time)

    if best_result is not None:
        print("\nBest run diagnostics:")
        print("Success           :", best_result.success)
        print("Message           :", best_result.message)
        print("Iterations        :", best_result.nit)
        print("Function evals    :", best_result.nfev)

    print("\n====================================================\n")