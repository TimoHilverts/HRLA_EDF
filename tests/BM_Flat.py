import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# MODEL:
# Retailer contracts (2):
#   (R1) flat: p
#   (R2) ToU : (l2_day, l2_night)
#
# Competitor contracts (1) -> outside option:
#   (C) flat: comp_flat
#
# Choice model: logit over 3 alternatives
#   {C, R1, R2} with P(j) ∝ exp(-beta * bill_j)
#
# Objective: U(x) = - expected retailer revenue
# Gradient: analytic (NumPy) + stable softmax
# Includes: ONE gradient check (finite differences) at competitor-like point
#
# Scaling:
#   z=(1,1,1)  <->  x=(p, l2_day, l2_night)=(comp_flat, comp_flat, comp_flat)
# ============================================================

S = 10
beta = 0.0069

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
# Competitor (ONLY flat)
# --------------------------
comp_flat = 0.342  # C: flat competitor

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
# x = (p, l2_day, l2_night)
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    p, l2_day, l2_night = map(float, x)

    profit = 0.0
    grad_profit = np.zeros(3, dtype=float)  # d(profit)/d(p, l2_day, l2_night)

    for s in range(S):
        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]
        T = E_day + E_night

        # Bills for 3 alternatives: [C_Flat, R1_Flat, R2_ToU]
        b_c  = comp_flat * T
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night
        b = np.array([b_c, b_r1, b_r2], dtype=float)

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)  # P[0]=C, P[1]=R1, P[2]=R2

        # Segment revenue (retailer only)
        r = b_r1 * P[1] + b_r2 * P[2]
        profit += w_s * r

        # Directional derivatives of b wrt retailer parameters
        # b = [b_c, b_r1, b_r2]
        db_dp = np.array([0.0, T, 0.0], dtype=float)              # affects only R1
        db_ld = np.array([0.0, 0.0, E_day], dtype=float)          # affects only R2
        db_ln = np.array([0.0, 0.0, E_night], dtype=float)        # affects only R2

        def dr_direction(db_dtheta: np.ndarray) -> float:
            # u = -beta*b  => du = -beta*db
            du = -beta * db_dtheta
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)  # directional derivative of P wrt theta

            # r = b_r1*P1 + b_r2*P2
            # dr = db_r1*P1 + b_r1*dP1 + db_r2*P2 + b_r2*dP2
            return (
                db_dtheta[1] * P[1] + b_r1 * dP[1]
                + db_dtheta[2] * P[2] + b_r2 * dP[2]
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
    return -g  # since U = -profit


# ============================================================
# Scaling wrapper (z-space) — "looks like the competitor"
# z=(1,1,1)  <->  x=(comp_flat, comp_flat, comp_flat)
# ============================================================
SCALE = np.array([
    comp_flat,  # scale p
    comp_flat,  # scale l2_day
    comp_flat,  # scale l2_night
], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    # Chain rule for x = SCALE*z
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
# Optimization + postprocessing (minimal)
# ============================================================
title = "BasicModel_N2_1competitor_flatonly_billlogit_uniformweights_scaled_numpy"
d = 3

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)                 # competitor-like start
    z0 += np.random.normal(0.0, 1, size=d)
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

    print("\nStarting Optimization (retailer: {R1 flat, R2 ToU}; competitor: flat only; bill-logit; uniform weights; scaled variables)")

    algorithm = GO.HRLA(
        d=d,
        M=50,
        N=1,
        K=20000,
        h=0.0001,
        title=title,
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[10],
        #[1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 20],
        sim_annealing=True,
    )

    print(f"Optimization finished. Samples saved to: {samples_filename}")
    print("NOTE: saved samples are z-vectors. Convert to real tariffs via x = to_x(z).")

    postprocessor = PostProcessor(samples_filename)

    measured = [20000]
   # [1, 3, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 7500, 10000, 12500, 14000, 15000,17500, 19000, 20000]
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)
