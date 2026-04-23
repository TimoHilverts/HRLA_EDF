import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# EXTENSION (NumPy gradient, no SymPy):
# - Parameter space d=3: (p, l2_day, l2_night)
# - Fixed Fees: REMOVED from both Retailer and Competitors
# - Consumption shifting: REMOVED (shift set to 0.0)
# ============================================================

S = 10
beta = 5 

# --------------------------
# Consumption (kWh): original
# --------------------------
Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

# shift = 0.0 effectively uses original Es
Es_used = Es_original.copy()

# --------------------------
# Segment weights
# --------------------------
w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161, 0.116, 0.045, 0.171, 0.112, 0.101], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# Competitors (Volumetric only)
# ============================================================
# A
tou_A  = np.array([0.355, 0.250], dtype=float)  # C3
flat_A = 0.342

# B
tou_B  = np.array([0.356, 0.255], dtype=float)  # C3
flat_B = 0.340

# No fixed fees (F=0)
competitors_np = np.array([
    [flat_A, flat_A],
    [tou_A[0], tou_A[1]],
    [flat_B, flat_B],
    [tou_B[0], tou_B[1]],
], dtype=float)

Es_np = np.array(Es_used, dtype=float)
comp_bills = Es_np @ competitors_np.T 
R_data = comp_bills.min(axis=1)

# ============================================================
# Profit + analytic gradient (3D x-space)
# x = (p, l2_day, l2_night)
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    p, l2_day, l2_night = map(float, x)
    profit = 0.0
    grad_profit = np.zeros(3, dtype=float) 

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s, R_s = float(w_data[s]), float(R_data[s])

        # Retailer bills (No fixed fees)
        b1 = p * T
        b2 = l2_day * E_day + l2_night * E_night
        
        DU = np.array([0.0, b1 - R_s, b2 - R_s], dtype=float)

        u = -beta * DU
        m = np.max(u)
        ex = np.exp(u - m)
        P = ex / np.sum(ex) # softmax_stable

        profit += w_s * (b1 * P[1] + b2 * P[2])

        def dr_direction(db1: float, db2: float) -> float:
            du = -beta * np.array([0.0, db1, db2], dtype=float)
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return db1 * P[1] + b1 * dP[1] + db2 * P[2] + b2 * dP[2]

        grad_profit[0] += w_s * dr_direction(db1=T, db2=0.0)      # dp
        grad_profit[1] += w_s * dr_direction(db1=0.0, db2=E_day) # dl2_day
        grad_profit[2] += w_s * dr_direction(db1=0.0, db2=E_night)# dl2_night

    return profit, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# ============================================================
# Scaling (3D)
# ============================================================
flat_scale = 0.5 * (flat_A + flat_B)
day_scale = 0.5 * (tou_A[0] + tou_B[0])
night_scale = 0.5 * (tou_A[1] + tou_B[1])

SCALE = np.array([flat_scale, day_scale, night_scale], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))


# ============================================================
# ONE gradient checker (finite differences) at competitor point
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
title = "Volumetric_d3_NoShift"
d = 3

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(loc=0.0, scale=0.05, size=d)
    return z0

if __name__ == "__main__":

    # ---- Gradient check at competitor-like x point (x = SCALE*1) ----
    x_test = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    print("\nStarting Optimization (N=2 retailer contracts; reservation bill outside; disutility-logit; data weights; shifted consumption; scaled variables)")

    algorithm = GO.HRLA(
        d=d,
        M=100,
        N=1,
        K=20000,
        h=0.000001,
        title=title,
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 20],
        sim_annealing=True,
    )

    print(f"Optimization finished. Samples saved to: {samples_filename}")
    print("NOTE: saved samples are z-vectors. Convert to real tariffs via x = to_x(z).")

    postprocessor = PostProcessor(samples_filename)

    measured = [1, 3, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 14000, 15000,
     17500, 19000, 20000]
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)
