# beta_sweep_baseline_tables_only.py
# ------------------------------------------------------------
# HABROK/HPC-friendly:
# For each beta:
#   - run HRLA (a fixed, K=5000, M=10)
#   - call compute_tables([5000], 1, "best") ONLY
# No get_best, no plotting, no extraction/parsing.
#
# Each beta run will save samples and a one-row table (at K=5000).
# You can postprocess/plot locally by reading those table files.
# ------------------------------------------------------------

import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor


# ============================================================
# DATA
# ============================================================
S = 10

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

# Competitors
comp_tou  = np.array([0.355, 0.250], dtype=float)  # C1: (day, night)
comp_flat = 0.342                                  # C2: flat

# Uniform weights
w_uniform = np.ones(S, dtype=float) / S


# ============================================================
# SCALING
# z=(1,1,1) <-> x=(p, l2_day, l2_night)=(comp_flat, comp_tou_day, comp_tou_night)
# ============================================================
SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)


# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)


# ============================================================
# Build (U_z, dU_z) closures for given beta
# ============================================================
def make_U_dU(beta_value: float):
    beta = float(beta_value)

    def profit_and_grad_x(x: np.ndarray) -> tuple[float, np.ndarray]:
        p, l2_day, l2_night = map(float, x)

        profit = 0.0
        grad_profit = np.zeros(3, dtype=float)

        for s in range(S):
            E_day, E_night = Es_used[s]
            w_s = w_uniform[s]
            T = E_day + E_night

            # Bills: [C1_ToU, C2_Flat, R1_Flat, R2_ToU]
            b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night
            b_c2 = comp_flat * T
            b_r1 = p * T
            b_r2 = l2_day * E_day + l2_night * E_night
            b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

            # Logit probabilities over 4 alternatives
            u = -beta * b
            P = softmax_stable(u)  # [C1, C2, R1, R2]

            # Retailer expected revenue (segment)
            profit += w_s * (b_r1 * P[2] + b_r2 * P[3])

            # Directional derivatives of bills wrt retailer parameters
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
        prof, _ = profit_and_grad_x(np.asarray(x, dtype=float))
        return -float(prof)  # U = -profit

    def dU_x(x: np.ndarray) -> np.ndarray:
        _, g = profit_and_grad_x(np.asarray(x, dtype=float))
        return -g

    def U_z(z: np.ndarray) -> float:
        return U_x(to_x(z))

    def dU_z(z: np.ndarray) -> np.ndarray:
        return SCALE * dU_x(to_x(z))

    return U_z, dU_z


# ============================================================
# Run HRLA + compute ONLY final table row (measured=[K])
# ============================================================
def run_one_beta_tables_only(beta: float, K: int, M: int, a_fixed: float, title_prefix: str, seed: int):
    np.random.seed(seed)

    d = 3
    U_z, dU_z = make_U_dU(beta)

    def initial() -> np.ndarray:
        z0 = np.ones(d, dtype=float)
        z0 += np.random.normal(0.0, 1.0, size=d)
        return z0

    algorithm = GO.HRLA(
        d=d,
        M=M,
        N=1,
        K=K,
        h=0.000001,
        title=f"{title_prefix}_beta{beta:g}",
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[a_fixed],
        sim_annealing=True,
    )

    postprocessor = PostProcessor(samples_filename)

    # ONE row only: final best value at K
    postprocessor.compute_tables([K], 1, "best")

    print(f"beta={beta:g} finished. Samples: {samples_filename} (tables computed at K={K})")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Fixed HRLA settings (as requested)
    a_fixed = 10.0
    K = 5000
    M = 10

    # β-grid (log-scale)
    betas = np.array([
    1e-4, 3e-4,
    7e-4, 9e-4, 1e-3, 1.2e-3,   # refined around the minimum
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    1.0,
    10.0
], dtype=float)

    title_prefix = "Baseline_N2_billlogit_uniformweights_scaled_numpy"

    for i, b in enumerate(betas):
        print(f"\n=== Running beta = {b:g} ===")
        run_one_beta_tables_only(
            beta=float(b),
            K=K,
            M=M,
            a_fixed=a_fixed,
            title_prefix=title_prefix,
            seed=1234 + i,
        )