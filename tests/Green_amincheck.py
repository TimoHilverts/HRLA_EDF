import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# EXTENSION: Retailer Green Options with Static Single Competitor
# - Retailer menu (N=4):
#   1) Normal Flat : (f1, p1)
#   2) Green Flat  : (f2, p2)
#   3) Normal ToU  : (f3, l3_day, l3_night)
#   4) Green ToU   : (f4, l4_day, l4_night)
# - Competitor: ONLY Competitor A (Flat + ToU) + Fixed Fee
# - Disutility DU_n = (bill_n - delta_n * g_s * B_reg_s) - R_s
# - Consumption shifting: 12% shift
# - Optimization: M=100, K=50000
# ============================================================

S = 10
beta = 5  

# --------------------------
# Consumption (kWh) + 12% shifting
# --------------------------
Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
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
# Segment weights (data)
# --------------------------
w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161, 0.116, 0.045, 0.171, 0.112, 0.101], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# STATIC SINGLE Competitor (A only)
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

competitor_np = np.array([
    [flat_A, flat_A],    # 0: A flat
    [tou_A[0], tou_A[1]] # 1: A ToU
], dtype=float)

fixed_fees_comp = np.array([F_A, F_A], dtype=float)

# Reservation bill R_s (min of A-flat and A-ToU)
Es_np = np.array(Es_used, dtype=float)
comp_var_bills = Es_np @ competitor_np.T 
comp_bills = comp_var_bills + fixed_fees_comp[None, :] 
R_data = comp_bills.min(axis=1) 

# Benchmark bill (B_reg) for green disutility credit
# Using average of Competitor A's contracts
lambda_reg = np.mean(competitor_np, axis=0)
f_reg = float(np.mean(fixed_fees_comp))
B_reg_data = f_reg + (Es_np @ lambda_reg.reshape(2, 1)).flatten()

# ============================================================
# Green component (retailer side)
# ============================================================
# Customer sensitivity to green contracts
g_levels = np.array([0.00, 0.04, 0.0, 0.04, 0.0, 0.02, 0.02, 0.00, 0.0, 0.04], dtype=float)
# Retailer contracts 2 and 4 (indices 1 and 3) are green
delta = np.array([0, 1, 0, 1], dtype=float)

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit (revenue) + analytic gradient (10D x-space)
# x = (f1,p1, f2,p2, f3,l3d,l3n, f4,l4d,l4n)
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float).flatten()
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

    revenue = 0.0
    grad_revenue = np.zeros(10, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s, R_s, g_s, B_reg_s = w_data[s], R_data[s], g_levels[s], B_reg_data[s]

        # Monetary Bills
        b1, b2 = f1 + p1 * T, f2 + p2 * T
        b3 = f3 + l3d * E_day + l3n * E_night
        b4 = f4 + l4d * E_day + l4n * E_night
        bills = np.array([b1, b2, b3, b4], dtype=float)

        # Disutilities (Outside + 4 retailer options)
        green_adjust = delta * g_s * B_reg_s
        DU_contracts = (bills - green_adjust) - R_s
        DU = np.concatenate(([0.0], DU_contracts))

        u = -beta * DU
        P = softmax_stable(u) 

        revenue += w_s * float(np.dot(bills, P[1:]))

        def dr_direction(dbills: np.ndarray) -> float:
            dbills = np.asarray(dbills, dtype=float).reshape(4,)
            dDU = np.concatenate(([0.0], dbills))
            du = -beta * dDU
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return float(np.dot(dbills, P[1:]) + np.dot(bills, dP[1:]))

        # Analytic gradients per parameter
        grad_revenue[0] += w_s * dr_direction([1.0, 0.0, 0.0, 0.0]) # f1
        grad_revenue[1] += w_s * dr_direction([T,   0.0, 0.0, 0.0]) # p1
        grad_revenue[2] += w_s * dr_direction([0.0, 1.0, 0.0, 0.0]) # f2
        grad_revenue[3] += w_s * dr_direction([0.0, T,   0.0, 0.0]) # p2
        grad_revenue[4] += w_s * dr_direction([0.0, 0.0, 1.0, 0.0]) # f3
        grad_revenue[5] += w_s * dr_direction([0.0, 0.0, E_day, 0.0]) # l3d
        grad_revenue[6] += w_s * dr_direction([0.0, 0.0, E_night, 0.0]) # l3n
        grad_revenue[7] += w_s * dr_direction([0.0, 0.0, 0.0, 1.0]) # f4
        grad_revenue[8] += w_s * dr_direction([0.0, 0.0, 0.0, E_day]) # l4d
        grad_revenue[9] += w_s * dr_direction([0.0, 0.0, 0.0, E_night]) # l4n

    return revenue, grad_revenue

def U_x(x: np.ndarray) -> float:
    rev, _ = profit_and_grad(x)
    return -float(rev)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(x)
    return -g

# ============================================================
# Scaling (Based on Competitor A)
# ============================================================
SCALE = np.array([
    F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]
], dtype=float)

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
# Optimization
# ============================================================
title = "ConShift_Green_SingleComp_FixedFees_numpy_exactgrad"
d = 10

z_test_points = np.random.uniform(0.5, 3.0, size=(1000, 10))
U_vals = np.array([U_z(z) for z in z_test_points])

for a_min in [0.1, 0.01, 0.001, 0.0001]:
    weights = np.exp(-a_min * U_vals)
    weights /= weights.sum()
    ess = 1 / np.sum(weights**2)
    print(f"a_min={a_min:.4f} | Max weight: {weights.max():.4f} | ESS: {ess:.1f}")