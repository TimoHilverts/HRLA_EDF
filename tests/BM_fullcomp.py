import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# BASIC MODEL (Fixed Fee for Competitor ONLY):
# Retailer contracts (2):
#   (R1) flat: p
#   (R2) ToU : (l2_day, l2_night)
#   NOTE: No fixed fees for retailer.
#
# Competitor (1 entity, 2 options):
#   (C1) ToU : (0.355, 0.250) + Fixed Fee (196.0)
#   (C2) flat: 0.342 + Fixed Fee (196.0)
#
# Choice model: logit over 4 alternatives {C1, C2, R1, R2}
# ============================================================

S = 10
beta = 5

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
# Competitor Setup (Static Environment)
# --------------------------
comp_tou = np.array([0.355, 0.250], dtype=float) 
comp_flat = 0.342
F_A = 196.0  # Fixed fee for COMPETITOR ONLY

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
    grad_profit = np.zeros(3, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]
        T = E_day + E_night

        # Bills for 4 alternatives
        # Competitor Bills include Fixed Fee F_A
        b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night + F_A
        b_c2 = comp_flat * T + F_A
        
        # Retailer Bills are Variable ONLY
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night
        
        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)  # P[0]=C1, P[1]=C2, P[2]=R1, P[3]=R2

        # Segment revenue (retailer only)
        r = b_r1 * P[2] + b_r2 * P[3]
        profit += w_s * r

        # Gradients wrt retailer parameters (p, l2_day, l2_night)
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

# ============================================================
# Optimization Utilities
# ============================================================
def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# Scaling (p, l2_day, l2_night)
SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))

# ============================================================
# Main Execution
# ============================================================
title = "Basic_RetailerVar_CompFixed"
d = 3

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(0.0, 0.5, size=d)
    return z0

if __name__ == "__main__":
    algorithm = GO.HRLA(
        d=d, M=100, N=1, K=50000, h=1e-6,
        title=title, U=U_z, dU=dU_z, initial=initial,
    )

    samples_filename = algorithm.generate_samples(As=[1, 3, 5, 10, 15, 20], sim_annealing=True)

    postprocessor = PostProcessor(samples_filename)
    measured = [1, 100, 1000, 5000, 10000, 20000, 30000, 40000, 50000]
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)
    
    print(f"\nOptimization complete. Final parameters (x-space): {to_x(bests[0])}")