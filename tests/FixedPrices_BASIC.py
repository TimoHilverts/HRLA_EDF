import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# EXTENSION: BasicModel with Fixed Fees (Retailer + Competitor)
# - Retailer contracts: R1 Flat (f1, p) and R2 ToU (f2, l2_day, l2_night)
# - Competitor contracts: C1 ToU and C2 Flat (both with Fixed Fee F_A)
# - Choice model: Logit over {C1, C2, R1, R2}
# - Weights: Uniform
# ============================================================

S = 10
beta = 0.003 

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
# Segment weights: NECESSARY CHANGE -> Uniform
# --------------------------
w_uniform = np.ones(S, dtype=float) / S

# ============================================================
# Competitor Environment (Static)
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float) 
flat_A = 0.342                                  
F_A    = 196.0 

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
        w_s = w_uniform[s]

        # NECESSARY CHANGE: Direct Bills for the 4-alternative logit
        b_c1 = F_A + tou_A[0] * E_day + tou_A[1] * E_night
        b_c2 = F_A + flat_A * T
        b_r1 = f1 + p * T
        b_r2 = f2 + l2_day * E_day + l2_night * E_night

        # NECESSARY CHANGE: Standard Logit over {C1, C2, R1, R2}
        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)
        u = -beta * b
        P = softmax_stable(u) 

        # Retailer revenue from R1 (index 2) and R2 (index 3)
        r = b_r1 * P[2] + b_r2 * P[3]
        profit += w_s * r

        def dr_direction(db_r1: float, db_r2: float) -> float:
            # db is derivative of bills w.r.t parameter theta
            # Only retailer bills change with retailer parameters
            db_dtheta = np.array([0.0, 0.0, db_r1, db_r2], dtype=float)
            du = -beta * db_dtheta
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return (
                db_dtheta[2] * P[2] + b_r1 * dP[2] + 
                db_dtheta[3] * P[3] + b_r2 * dP[3]
            )

        grad_profit[0] += w_s * dr_direction(db_r1=1.0, db_r2=0.0)    # df1
        grad_profit[1] += w_s * dr_direction(db_r1=T,   db_r2=0.0)    # dp
        grad_profit[2] += w_s * dr_direction(db_r1=0.0, db_r2=1.0)    # df2
        grad_profit[3] += w_s * dr_direction(db_r1=0.0, db_r2=E_day)  # dl2_day
        grad_profit[4] += w_s * dr_direction(db_r1=0.0, db_r2=E_night)# dl2_night

    return profit, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g 

# ============================================================
# Scaling (Same as before)
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
title = "BasicModel_FixedFees_Uniform_Numpy"
d = 5

def initial() -> np.ndarray:
    return np.random.uniform(low=0.5, high=3.0, size=d)

def make_constant_h(h_val):
    def constant_h(k, K):
        return h_val
    return constant_h

if __name__ == "__main__":
    x_test = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    algorithm = GO.HRLA(
        d=d, M=50, N=1, K=10000,
        h=make_constant_h(1e-4),
        a_min=0.1,
        #power=2,
        title=title, U=U_z, dU=dU_z, initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[1, 3, 5, 10, 15, 20], 
        sim_annealing=True,
    )

    postprocessor = PostProcessor(samples_filename)
    measured = [10000]
    postprocessor.compute_tables(measured, 1, "mean")
    postprocessor.compute_tables(measured, 1, "std")
    postprocessor.compute_tables(measured, 1, "best")