# beta_sweep_complete_model_tables_only.py
# ------------------------------------------------------------
# HABROK/HPC-friendly:
# For each beta:
#   - run HRLA for the complete model
#   - call compute_tables([K], 1, "best") ONLY
# No get_best, no plotting, no extraction/parsing.
#
# COMPLETE MODEL:
#   Retailer menu (N=4):
#     1) Normal Flat : (f1, p1)
#     2) Green Flat  : (f2, p2)
#     3) Normal ToU  : (f3, l3_day, l3_night)
#     4) Green ToU   : (f4, l4_day, l4_night)
#
#   Competitor:
#     - Single static competitor A
#     - Flat + ToU + fixed fee
#
#   Features included:
#     - Segment weights
#     - Reservation bill
#     - Green disutility adjustment
#     - 12% consumption shifting
#
#   Output:
#     - Final "best" table row at iteration K for each beta
# ------------------------------------------------------------

import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor


# ============================================================
# DATA
# ============================================================
S = 10

# --------------------------
# Original consumption (kWh)
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


# ============================================================
# CONSUMPTION SHIFTING
# ============================================================
def shifting_cons(E: np.ndarray, shift: float) -> np.ndarray:
    E_day, E_night = E
    E_day_new = (1.0 - shift) * E_day
    E_night_new = E_night + shift * E_day
    return np.array([E_day_new, E_night_new], dtype=float)


shift = 0.12
Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)


# ============================================================
# SEGMENT WEIGHTS
# ============================================================
w_data = np.array(
    [0.125, 0.057, 0.068, 0.044, 0.161, 0.116, 0.045, 0.171, 0.112, 0.101],
    dtype=float
)
w_data = w_data / np.sum(w_data)


# ============================================================
# STATIC SINGLE COMPETITOR A
# ============================================================
tou_A = np.array([0.355, 0.250], dtype=float)  # (day, night)
flat_A = 0.342
F_A = 196.0

competitor_np = np.array([
    [flat_A, flat_A],      # A flat
    [tou_A[0], tou_A[1]],  # A ToU
], dtype=float)

fixed_fees_comp = np.array([F_A, F_A], dtype=float)

# Reservation bill R_s = min(A-flat, A-ToU)
Es_np = np.array(Es_used, dtype=float)
comp_var_bills = Es_np @ competitor_np.T
comp_bills = comp_var_bills + fixed_fees_comp[None, :]
R_data = comp_bills.min(axis=1)

# Benchmark bill for green disutility credit
lambda_reg = np.mean(competitor_np, axis=0)
f_reg = float(np.mean(fixed_fees_comp))
B_reg_data = f_reg + (Es_np @ lambda_reg.reshape(2, 1)).flatten()


# ============================================================
# GREEN PREFERENCES
# ============================================================
g_levels = np.array(
    [0.00, 0.04, 0.0, 0.04, 0.0, 0.02, 0.02, 0.00, 0.0, 0.04],
    dtype=float
)

# Retailer contracts 2 and 4 are green
delta = np.array([0, 1, 0, 1], dtype=float)


# ============================================================
# STABLE SOFTMAX
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)


# ============================================================
# SCALING
# x = (f1,p1, f2,p2, f3,l3d,l3n, f4,l4d,l4n)
# z scaled relative to competitor A
# ============================================================
SCALE = np.array([
    F_A, flat_A,
    F_A, flat_A,
    F_A, tou_A[0], tou_A[1],
    F_A, tou_A[0], tou_A[1]
], dtype=float)


def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)


# ============================================================
# BUILD (U_z, dU_z) FOR GIVEN beta
# ============================================================
def make_U_dU(beta_value: float):
    beta = float(beta_value)

    def profit_and_grad_x(x: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=float).flatten()
        f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

        revenue = 0.0
        grad_revenue = np.zeros(10, dtype=float)

        for s in range(S):
            E_day, E_night = Es_used[s]
            T = E_day + E_night
            w_s = w_data[s]
            R_s = R_data[s]
            g_s = g_levels[s]
            B_reg_s = B_reg_data[s]

            # Retailer bills
            b1 = f1 + p1 * T
            b2 = f2 + p2 * T
            b3 = f3 + l3d * E_day + l3n * E_night
            b4 = f4 + l4d * E_day + l4n * E_night
            bills = np.array([b1, b2, b3, b4], dtype=float)

            # Disutilities: outside + 4 retailer options
            green_adjust = delta * g_s * B_reg_s
            DU_contracts = (bills - green_adjust) - R_s
            DU = np.concatenate(([0.0], DU_contracts))  # outside option has disutility 0

            u = -beta * DU
            P = softmax_stable(u)  # [outside, R1, R2, R3, R4]

            # Expected retailer revenue
            revenue += w_s * float(np.dot(bills, P[1:]))

            def dr_direction(dbills: np.ndarray) -> float:
                dbills = np.asarray(dbills, dtype=float).reshape(4,)
                dDU = np.concatenate(([0.0], dbills))
                du = -beta * dDU
                du_bar = float(np.dot(P, du))
                dP = P * (du - du_bar)
                return float(np.dot(dbills, P[1:]) + np.dot(bills, dP[1:]))

            # Gradient entries
            grad_revenue[0] += w_s * dr_direction([1.0, 0.0, 0.0, 0.0])       # f1
            grad_revenue[1] += w_s * dr_direction([T,   0.0, 0.0, 0.0])       # p1
            grad_revenue[2] += w_s * dr_direction([0.0, 1.0, 0.0, 0.0])       # f2
            grad_revenue[3] += w_s * dr_direction([0.0, T,   0.0, 0.0])       # p2
            grad_revenue[4] += w_s * dr_direction([0.0, 0.0, 1.0, 0.0])       # f3
            grad_revenue[5] += w_s * dr_direction([0.0, 0.0, E_day, 0.0])     # l3_day
            grad_revenue[6] += w_s * dr_direction([0.0, 0.0, E_night, 0.0])   # l3_night
            grad_revenue[7] += w_s * dr_direction([0.0, 0.0, 0.0, 1.0])       # f4
            grad_revenue[8] += w_s * dr_direction([0.0, 0.0, 0.0, E_day])     # l4_day
            grad_revenue[9] += w_s * dr_direction([0.0, 0.0, 0.0, E_night])   # l4_night

        return revenue, grad_revenue

    def U_x(x: np.ndarray) -> float:
        rev, _ = profit_and_grad_x(np.asarray(x, dtype=float))
        return -float(rev)

    def dU_x(x: np.ndarray) -> np.ndarray:
        _, g = profit_and_grad_x(np.asarray(x, dtype=float))
        return -g

    def U_z(z: np.ndarray) -> float:
        return U_x(to_x(z))

    def dU_z(z: np.ndarray) -> np.ndarray:
        return SCALE * dU_x(to_x(z))

    return U_z, dU_z


# ============================================================
# RUN HRLA + COMPUTE ONLY FINAL "BEST" TABLE
# ============================================================
def run_one_beta_tables_only(beta: float, K: int, M: int, a_fixed: float, title_prefix: str, seed: int):
    np.random.seed(seed)

    d = 10
    U_z, dU_z = make_U_dU(beta)

    def initial() -> np.ndarray:
        # Same range as your complete model
        return np.random.uniform(low=0.5, high=3.0, size=d)

    algorithm = GO.HRLA(
        d=d,
        M=M,
        N=1,
        K=K,
        h=1e-4,
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

    # One row only: final best value at K
    postprocessor.compute_tables([K], 1, "best")

    print(f"beta={beta:g} finished. Samples: {samples_filename} (tables computed at K={K})")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Fixed HRLA settings
    # You can change a_fixed if another a performed better in your tests.
    a_fixed = 10.0
    K = 10000
    M = 10

    # Beta grid
    betas = np.array([
        1e-4, 3e-4,
        7e-4, 9e-4, 1e-3, 1.2e-3,
        2e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        1.0,
        10.0
    ], dtype=float)

    title_prefix = "CompleteModel_Green_SingleComp_FixedFees_Shifted_Weighted"

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