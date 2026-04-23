import numpy as np
import time
from scipy.optimize import differential_evolution

# ============================================================
# EXTENSION: BasicModel with Fixed Fees (Retailer + Competitor)
# - Retailer contracts: R1 Flat (f1, p) and R2 ToU (f2, l2_day, l2_night)
# - Competitor contracts: C1 ToU and C2 Flat (both with Fixed Fee F_A)
# - Choice model: Logit over {C1, C2, R1, R2}
# - Weights: Uniform
# - Optimization: Differential Evolution
# ============================================================

S = 10
beta = 0.03

# --------------------------
# Consumption (kWh): original
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
# Segment weights: uniform
# --------------------------
w_uniform = np.ones(S, dtype=float) / S

# ============================================================
# Competitor Environment (Static)
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit function
# x = [f1, p, f2, l2_day, l2_night]
# ============================================================
def profit(x: np.ndarray) -> float:
    f1, p, f2, l2_day, l2_night = map(float, x)
    total_profit = 0.0

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s = w_uniform[s]

        # Bills
        b_c1 = F_A + tou_A[0] * E_day + tou_A[1] * E_night
        b_c2 = F_A + flat_A * T
        b_r1 = f1 + p * T
        b_r2 = f2 + l2_day * E_day + l2_night * E_night

        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)

        # Retailer expected revenue
        r = b_r1 * P[2] + b_r2 * P[3]
        total_profit += w_s * r

    return total_profit

# ============================================================
# Objective for minimizer
# ============================================================
def objective_x(x: np.ndarray) -> float:
    return -profit(x)

# ============================================================
# Scaling (same idea as HRLA)
# z = scaled variables, x = real tariff variables
# ============================================================
SCALE = np.array([F_A, flat_A, F_A, tou_A[0], tou_A[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def objective_z(z: np.ndarray) -> float:
    return objective_x(to_x(z))

# ============================================================
# Optional gradient checker (can be removed if not needed)
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    f1, p, f2, l2_day, l2_night = map(float, x)
    profit_val = 0.0
    grad_profit = np.zeros(5, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s = w_uniform[s]

        b_c1 = F_A + tou_A[0] * E_day + tou_A[1] * E_night
        b_c2 = F_A + flat_A * T
        b_r1 = f1 + p * T
        b_r2 = f2 + l2_day * E_day + l2_night * E_night

        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)
        u = -beta * b
        P = softmax_stable(u)

        r = b_r1 * P[2] + b_r2 * P[3]
        profit_val += w_s * r

        def dr_direction(db_r1: float, db_r2: float) -> float:
            db_dtheta = np.array([0.0, 0.0, db_r1, db_r2], dtype=float)
            du = -beta * db_dtheta
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return (
                db_dtheta[2] * P[2] + b_r1 * dP[2] +
                db_dtheta[3] * P[3] + b_r2 * dP[3]
            )

        grad_profit[0] += w_s * dr_direction(db_r1=1.0,   db_r2=0.0)      # df1
        grad_profit[1] += w_s * dr_direction(db_r1=T,     db_r2=0.0)      # dp
        grad_profit[2] += w_s * dr_direction(db_r1=0.0,   db_r2=1.0)      # df2
        grad_profit[3] += w_s * dr_direction(db_r1=0.0,   db_r2=E_day)    # dl2_day
        grad_profit[4] += w_s * dr_direction(db_r1=0.0,   db_r2=E_night)  # dl2_night

    return profit_val, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g

# ============================================================
# Differential Evolution bounds in scaled space z
# x = SCALE * z
#
# These are the direct analogues of the earlier model:
# f1       in [0.5*F_A,   2.0*F_A]
# p        in [0.5*flat_A,2.0*flat_A]
# f2       in [0.5*F_A,   2.0*F_A]
# l2_day   in [0.5*tou_A[0], 2.0*tou_A[0]]
# l2_night in [0.5*tou_A[1], 2.5*tou_A[1]]
# ============================================================
bounds = [
    (0, 3.0),   # z_f1
    (0, 3.0),   # z_p
    (0, 3.0),   # z_f2
    (0, 3.0),   # z_l2_day
    (0, 3),   # z_l2_night
]

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Optional gradient check
    x_test = to_x(np.ones(5, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    # --------------------------
    # Differential Evolution runs
    # --------------------------
    N_RUNS = 20

    revenues = []
    runtimes = []

    best_overall = -np.inf
    best_x = None
    best_z = None

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

        z_best = result.x
        x_best = to_x(z_best)
        revenue = profit(x_best)

        revenues.append(revenue)
        runtimes.append(end - start)

        if revenue > best_overall:
            best_overall = revenue
            best_x = x_best.copy()
            best_z = z_best.copy()

        print(f"Run {i+1:2d}: revenue = {revenue:.6f}")

    revenues = np.array(revenues)
    runtimes = np.array(runtimes)

    mean_rev = np.mean(revenues)
    std_rev = np.std(revenues)

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
    print("\nMean revenue      :", mean_rev)
    print("Std revenue       :", std_rev)

    print("\nMean runtime (s)  :", mean_time)
    print("Std runtime (s)   :", std_time)

    print("\n====================================================\n")