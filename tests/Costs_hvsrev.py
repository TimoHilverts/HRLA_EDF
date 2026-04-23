import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# DATA & PARAMETERS
# ============================================================
S     = 10
beta  = 0.02
shift = 0.12

C_DAY   = 0.29
C_NIGHT = 0.21
C_GREEN = 0.010
C_SERVE = 110.0

Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

def shifting_cons(E: np.ndarray, s: float) -> np.ndarray:
    return np.array([(1.0 - s) * E[0], E[1] + s * E[0]], dtype=float)

Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161,
                    0.116, 0.045, 0.171, 0.112, 0.101], dtype=float)
w_data /= np.sum(w_data)

def get_avg_cost():
    total_day   = np.sum(Es_original[:, 0])
    total_night = np.sum(Es_original[:, 1])
    return (total_day * C_DAY + total_night * C_NIGHT) / (total_day + total_night)

C_FLAT = get_avg_cost()

tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

competitor_np   = np.array([[flat_A, flat_A], [tou_A[0], tou_A[1]]], dtype=float)
fixed_fees_comp = np.array([F_A, F_A], dtype=float)

comp_bills = (Es_used @ competitor_np.T) + fixed_fees_comp[None, :]
R_data     = comp_bills.min(axis=1)

lambda_reg = np.mean(competitor_np, axis=0)
B_reg_data = F_A + (Es_used @ lambda_reg)

g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00,
                     0.02, 0.02, 0.00, 0.00, 0.04], dtype=float)
delta    = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

def cost_per_contract(E_day: float, E_night: float) -> np.ndarray:
    T = E_day + E_night
    c_tou_base  = C_SERVE + (C_DAY * E_day + C_NIGHT * E_night)
    c_flat_base = C_SERVE + (C_FLAT * T)
    return np.array([
        c_flat_base,
        c_flat_base + (C_GREEN * T),
        c_tou_base,
        c_tou_base + (C_GREEN * T),
    ], dtype=float)

SCALE = np.array([
    F_A, flat_A, F_A, flat_A,
    F_A, tou_A[0], tou_A[1],
    F_A, tou_A[0], tou_A[1]
], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float).flatten()
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

    total_profit = 0.0
    grad_profit  = np.zeros(10, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T       = E_day + E_night
        w_s     = w_data[s]
        R_s     = R_data[s]
        g_s     = g_levels[s]
        B_reg_s = B_reg_data[s]

        b1 = f1 + p1 * T
        b2 = f2 + p2 * T
        b3 = f3 + l3d * E_day + l3n * E_night
        b4 = f4 + l4d * E_day + l4n * E_night
        bills = np.array([b1, b2, b3, b4], dtype=float)

        costs   = cost_per_contract(E_day, E_night)
        margins = bills - costs

        green_adjust = delta * g_s * B_reg_s
        DU_contracts = (bills - green_adjust) - R_s
        DU = np.concatenate(([0.0], DU_contracts))

        u = -beta * DU
        P = softmax_stable(u)

        total_profit += w_s * float(np.dot(margins, P[1:]))

        def dp_direction(dbills: np.ndarray) -> float:
            dbills = np.asarray(dbills, dtype=float).reshape(4,)
            dDU    = np.concatenate(([0.0], dbills))
            du     = -beta * dDU
            du_bar = float(np.dot(P, du))
            dP     = P * (du - du_bar)
            return float(np.dot(dbills, P[1:]) + np.dot(margins, dP[1:]))

        grad_profit[0] += w_s * dp_direction([1.0, 0.0, 0.0, 0.0])
        grad_profit[1] += w_s * dp_direction([T,   0.0, 0.0, 0.0])
        grad_profit[2] += w_s * dp_direction([0.0, 1.0, 0.0, 0.0])
        grad_profit[3] += w_s * dp_direction([0.0, T,   0.0, 0.0])
        grad_profit[4] += w_s * dp_direction([0.0, 0.0, 1.0, 0.0])
        grad_profit[5] += w_s * dp_direction([0.0, 0.0, E_day,   0.0])
        grad_profit[6] += w_s * dp_direction([0.0, 0.0, E_night, 0.0])
        grad_profit[7] += w_s * dp_direction([0.0, 0.0, 0.0, 1.0])
        grad_profit[8] += w_s * dp_direction([0.0, 0.0, 0.0, E_day])
        grad_profit[9] += w_s * dp_direction([0.0, 0.0, 0.0, E_night])

    return total_profit, grad_profit

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(x)
    return -float(prof)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(x)
    return -g

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))

def initial() -> np.ndarray:
    return np.random.uniform(low=0.0, high=2.0, size=10)

# ============================================================
# MAIN EXECUTION — h sensitivity analysis
# ============================================================
d       = 10
A_FIXED = 10      # fixed amax — moderate so h differences are visible
K       = 10000     # small K for the same reason
M       = 100

H_VALUES = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

if __name__ == "__main__":

    # Run one HRLA instance per h value, collect filenames
    filenames = []

    for h_val in H_VALUES:
        print(f"Running h={h_val:.0e} ...")

        algorithm = GO.HRLA(
            d=d, M=M, N=1, K=K, h=h_val,
            a_min=0.1,
            title=f"h_sensitivity_{h_val:.0e}",
            U=U_z, dU=dU_z, initial=initial,
        )

        filename = algorithm.generate_samples(
            As=[A_FIXED], sim_annealing=True
        )
        filenames.append(filename)

    # Plot convergence curves — one curve per h, using existing method
    PostProcessor.plot_running_best_revenue_for_h(
        data_filenames=filenames,
        hs=H_VALUES,
        dpi=50,
        a_idx=0,          # As=[A_FIXED] so only index 0 exists
        U=U_z,
        dU=dU_z,
    )

    # # Summary table — mean, std, best revenue per h
    # PostProcessor.compare_h_table(
    #     data_filenames=filenames,
    #     a_idx=0,
    #     dpi=1,
    #     U=U_z,
    #     dU=dU_z,
    # )