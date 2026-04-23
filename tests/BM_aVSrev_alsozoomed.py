import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# MODEL:
# Retailer contracts (2):
#   (R1) flat: p
#   (R2) ToU : (l2_day, l2_night)
#
# Competitor contracts (2) -> outside options:
#   (C1) ToU : (0.37, 0.28) + Fixed Fee 196
#   (C2) flat: 0.345 + Fixed Fee 196
# ============================================================

S = 10
beta = 0.01
COMP_FIXED_FEE = 196.0  # Added fixed fee constant

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
comp_tou = np.array([0.355, 0.250], dtype=float)   # C1: (day, night)
comp_flat = 0.342                                  # C2: flat

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
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    p, l2_day, l2_night = map(float, x)

    profit = 0.0
    grad_profit = np.zeros(3, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]
        T = E_day + E_night

        # Bills for 4 alternatives
        # Added COMP_FIXED_FEE to both competitor options
        b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night + COMP_FIXED_FEE
        b_c2 = comp_flat * T + COMP_FIXED_FEE
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night
        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)

        # Segment revenue (retailer only)
        r = b_r1 * P[2] + b_r2 * P[3]
        profit += w_s * r

        # Directional derivatives (Retailer parameters only)
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
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# ============================================================
# Scaling wrapper (z-space)
# ============================================================
SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

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
# Optimization Execution
# ============================================================
title = "BasicModel_N2_2competitors_FixedFee_billlogit_uniformweights_scaled_numpy"
d = 3

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(0.0, 1 , size=d)
    return z0

if __name__ == "__main__":
    x_test = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    algorithm = GO.HRLA(
        d=d, M=10, N=1, K=10000, h=0.0001,
        title=title, U=U_z, dU=dU_z, initial=initial,
    )

    samples_filename = algorithm.generate_samples(As=[0.1, 1, 5, 10, 20, 40, 80, 100], sim_annealing=False)
    postprocessor = PostProcessor(samples_filename)

    postprocessor.plot_running_best_revenue_zoom(dpi=50, zoom_K=300)
    #postprocessor.print_best_across_all_as(dpi=1, to_x=to_x)
  