import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# EXTENSION: Single Competitor + Consumption Shifting
# - Retailer contracts: Flat (f1, p) and ToU (f2, l2_day, l2_night)
# - Static Competitive Environment: Competitor A only (with Fixed Fee)
# - Consumption shifting: 12% shift from day to night
# - Choice model: logit over {outside, R1, R2}
# - Optimization: M=100, K=50000
# ============================================================

S = 10
beta = 0.003  

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

shift = 0.12 # Extension: 12% shifting
Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

# --------------------------
# Segment weights (empirical data)
# --------------------------
w_data = np.array([
    0.125, 0.057, 0.068, 0.044, 0.161,
    0.116, 0.045, 0.171, 0.112, 0.101
], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# SINGLE Competitor (Environment)
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float) 
flat_A = 0.342                                  
F_A    = 196.0

# Define competitor tariffs (Flat and ToU)
competitor_np = np.array([
    [flat_A, flat_A],    
    [tou_A[0], tou_A[1]] 
], dtype=float)

fixed_fees_comp = np.array([F_A, F_A], dtype=float)

Es_np = np.array(Es_used, dtype=float)
comp_var_bills = Es_np @ competitor_np.T 
comp_bills = comp_var_bills + fixed_fees_comp[None, :] 

# Reservation bill is the best option between Competitor A's Flat and ToU
R_data = comp_bills.min(axis=1) 

# ============================================================
# Stable softmax & Profit Logic
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    f1, p, f2, l2_day, l2_night = map(float, x)
    profit = 0.0
    grad_profit = np.zeros(5, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s, R_s = float(w_data[s]), float(R_data[s])

        b1 = f1 + p * T
        b2 = f2 + l2_day * E_day + l2_night * E_night

        DU = np.array([0.0, b1 - R_s, b2 - R_s], dtype=float)
        u = -beta * DU
        P = softmax_stable(u) 

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
        grad_profit[2] += w_s * dr_direction(db1=0.0, db2=1.0)      # f2
        grad_profit[3] += w_s * dr_direction(db1=0.0, db2=E_day)    # l2_day
        grad_profit[4] += w_s * dr_direction(db1=0.0, db2=E_night)  # l2_night

    return profit, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g 

# ============================================================
# Scaling (Based on Competitor A values)
# ============================================================
SCALE = np.array([F_A, flat_A, F_A, tou_A[0], tou_A[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))

# ============================================================
# ONE gradient checker (finite differences)
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
title = "ConShift_SingleComp_FixedFees_RetailerFixedFees_Numpy"
d = 5

def initial() -> np.ndarray:
    return np.ones(d, dtype=float) + np.random.normal(0, 0.2, d)

if __name__ == "__main__":
    # ---- Gradient check at competitor-like x point ----
    x_test = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    print(f"\nStarting Optimization (M=100, K=50000, Shift={shift}, Comp Fixed Fee={F_A})")

    algorithm = GO.HRLA(
        d=d, M=50, N=1, K=50000, h=1e-6,
        title=title, U=U_z, dU=dU_z, initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[1, 3, 5, 10, 15, 20], 
        sim_annealing=True,
    )

    postprocessor = PostProcessor(samples_filename)
    measured = [1, 10000, 30000, 50000]
    
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)