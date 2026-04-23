import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# EXTENSION (NumPy gradient, no SymPy):
# - Retailer contracts (2):
#   (R1) Flat:   (f1, p)      => bill = f1 + p*(E_day + E_night)
#   (R2) ToU:    (f2, l2_day, l2_night)
#               => bill = f2 + l2_day*E_day + l2_night*E_night
#
# - Customers face an outside option defined by reservation bill:
#   R_s = min over competitor TOTAL bills (variable + fixed)
#
# - Disutility formulation (as in your SymPy model):
#   DU_outside = 0
#   DU_Rn      = bill_Rn - R_s
#
# - Choice model: logit over {outside, R1, R2}
#     P(j) ∝ exp(-beta * DU_j)
#
# - Objective: U(x) = - expected retailer revenue
# - Gradient: analytic (NumPy) + stable softmax
# - Includes: ONE gradient check (finite differences) at competitor-like point
#
# - Consumption shifting: Es_used derived from Es_original
# ============================================================

S = 10
beta = 5  # keep your extension value (change if desired)

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

shift = 0.15
Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

# --------------------------
# Segment weights (data)
# --------------------------
w_data = np.array([
    0.125, 0.057, 0.068, 0.044, 0.161,
    0.116, 0.045, 0.171, 0.112, 0.101
], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# Competitors (used ONLY to build reservation bills R_s)
# Competitor A: use the SAME 2 tariffs as the basic model
#   - ToU:  (0.37, 0.20)
#   - Flat: 0.345
# Both get the same annual fixed fee F_A (as in your extension structure).
# ============================================================

tou_A  = np.array([0.37, 0.28], dtype=float)
flat_A = np.array([0.345, 0.345], dtype=float)
F_A    = 160.0

tou_B  = np.array([0.35, 0.30], dtype=float)
flat_B = np.array([0.330, 0.330], dtype=float)
F_B    = 152.0

competitors = [flat_A, tou_A, flat_B, tou_B]
fixed_fees_comp_list = [F_A, F_A, F_B, F_B]


competitors_np = np.array(competitors, dtype=float)   # shape (4, 2)
Es_np = np.array(Es_used, dtype=float)                # shape (S, 2)

comp_var_bills = Es_np @ competitors_np.T             # shape (S, 4)
comp_fixed = np.array(fixed_fees_comp_list, dtype=float)  # shape (4,)
comp_bills = comp_var_bills + comp_fixed[None, :]     # shape (S, 4)
R_data = comp_bills.min(axis=1)                       # shape (S,)

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit + analytic gradient (x-space)
# x = (f1, p, f2, l2_day, l2_night)
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    f1, p, f2, l2_day, l2_night = map(float, x)

    profit = 0.0
    grad_profit = np.zeros(5, dtype=float)  # d(profit)/d(f1, p, f2, l2_day, l2_night)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s = float(w_data[s])
        R_s = float(R_data[s])

        # Retailer bills
        b1 = f1 + p * T
        b2 = f2 + l2_day * E_day + l2_night * E_night

        # Disutilities (outside has DU=0)
        # DU1 = b1 - R_s, DU2 = b2 - R_s
        DU = np.array([0.0, b1 - R_s, b2 - R_s], dtype=float)

        # Logit probabilities over {outside, R1, R2}
        u = -beta * DU
        P = softmax_stable(u)  # P[0]=outside, P[1]=R1, P[2]=R2

        # Segment expected retailer revenue
        r = b1 * P[1] + b2 * P[2]
        profit += w_s * r

        # Helper: directional derivative dr/dtheta, where theta affects (b1,b2)
        def dr_direction(db1: float, db2: float) -> float:
            # DU = [0, b1-R, b2-R] so dDU = [0, db1, db2]
            dDU = np.array([0.0, db1, db2], dtype=float)

            # u = -beta*DU => du = -beta*dDU
            du = -beta * dDU
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)

            # r = b1*P1 + b2*P2
            # dr = db1*P1 + b1*dP1 + db2*P2 + b2*dP2
            return db1 * P[1] + b1 * dP[1] + db2 * P[2] + b2 * dP[2]

        # Derivatives of (b1,b2) wrt each parameter
        # f1: b1+=1
        grad_profit[0] += w_s * dr_direction(db1=1.0, db2=0.0)

        # p: b1+=T
        grad_profit[1] += w_s * dr_direction(db1=T, db2=0.0)

        # f2: b2+=1
        grad_profit[2] += w_s * dr_direction(db1=0.0, db2=1.0)

        # l2_day: b2+=E_day
        grad_profit[3] += w_s * dr_direction(db1=0.0, db2=E_day)

        # l2_night: b2+=E_night
        grad_profit[4] += w_s * dr_direction(db1=0.0, db2=E_night)

    return profit, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g  # since U = -profit

# ============================================================
# Scaling (REPRESENT THE TWO COMPETITORS explicitly)
# - Fixed fees: average of F_A and F_B
# - Flat price: average of competitor flat tariffs (A and B)
# - ToU prices: average of competitor ToU tariffs (A and B), per time period
# ============================================================

fixed_fee_scale = 0.5 * (F_A + F_B)

flat_price_A = float(flat_A[0])   # = 0.345
flat_price_B = float(flat_B[0])   # = 0.330
flat_scale = 0.5 * (flat_price_A + flat_price_B)

tou_day_scale   = 0.5 * (float(tou_A[0]) + float(tou_B[0]))
tou_night_scale = 0.5 * (float(tou_A[1]) + float(tou_B[1]))

SCALE = np.array([
    fixed_fee_scale,  # f1
    flat_scale,       # p
    fixed_fee_scale,  # f2
    tou_day_scale,    # l2_day
    tou_night_scale,  # l2_night
], dtype=float)


def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    # Chain rule for x = SCALE*z
    return SCALE * dU_x(to_x(z))

# ============================================================
# ONE gradient checker (finite differences) at competitor-like point
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
# Optimization + postprocessing (minimal)
# ============================================================
title = "ConShift_numpy_exactgrad"
d = 5

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(loc=0.0, scale=0.1, size=d)
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
        M=10,
        N=1,
        K=30000,
        h=0.000001,
        title=title,
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[0.1, 0.5, 1, 2, 3, 4, 5, 10, 15, 20],
        sim_annealing=True,
    )

    print(f"Optimization finished. Samples saved to: {samples_filename}")
    print("NOTE: saved samples are z-vectors. Convert to real tariffs via x = to_x(z).")

    postprocessor = PostProcessor(samples_filename)

    measured = [1, 3, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000,
     3000, 4000, 5000, 7500, 10000, 12500, 14000, 15000, 17500, 19000, 20000, 23000, 26000, 28000, 30000]
     #, 40000, 45000, 47000, 49000, 50000]
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)
