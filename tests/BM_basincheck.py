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
beta = 5
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

# def initial() -> np.ndarray:
#     """
#     Spreads 100 agents across a range that includes the 
#     global optimum z-values (approx 1.12, 0.97, 2.42).
#     """
#     # Low bound 0.5, High bound 3.0 to catch that 2.42 Night Price
#     return np.random.uniform(low=0.5, high=3.0, size=d)

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(1, 2 , size=d)
    return z0

z_star = np.array([1.127, 0.977, 2.428])
U_star = U_z(z_star)
revenue_star = -U_star

print(f"Global optimum: revenue={revenue_star:.4f} at z={z_star}")
print()

# How much revenue drop do we consider significant?
drop_threshold = 1.0  # 1 unit of revenue

print("Basin width scan per dimension:")
print("=" * 60)

for dim in range(d):
    print(f"\nDimension {dim} (z[{dim}] = {z_star[dim]:.3f}):")
    
    # Scan positive and negative directions
    for direction, sign in [("positive", 1), ("negative", -1)]:
        for delta in [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]:
            z_test = z_star.copy()
            z_test[dim] += sign * delta
            revenue_test = -U_z(z_test)
            drop = revenue_star - revenue_test
            marker = " ← basin edge" if drop > drop_threshold else ""
            print(f"  {direction} delta={delta:.3f} | revenue={revenue_test:.4f} | drop={drop:.4f}{marker}")