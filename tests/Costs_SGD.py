import numpy as np
import time

# ============================================================
# DATA & PARAMETERS (Full Model - Matching HRLA)
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

def shifting_cons(E, s):
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
comp_bills      = (Es_used @ competitor_np.T) + fixed_fees_comp[None, :]
R_data          = comp_bills.min(axis=1)

lambda_reg = np.mean(competitor_np, axis=0)
B_reg_data = F_A + (Es_used @ lambda_reg)

g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00,
                     0.02, 0.02, 0.00, 0.00, 0.04], dtype=float)
delta    = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

SCALE = np.array([F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]], dtype=float)

# ============================================================
# COST FUNCTION (matching HRLA)
# ============================================================
def cost_per_contract(E_day, E_night):
    T = E_day + E_night
    c_tou_base  = C_SERVE + (C_DAY * E_day + C_NIGHT * E_night)
    c_flat_base = C_SERVE + (C_FLAT * T)
    return np.array([
        c_flat_base,
        c_flat_base + (C_GREEN * T),
        c_tou_base,
        c_tou_base + (C_GREEN * T),
    ], dtype=float)

# ============================================================
# PROFIT & ANALYTIC GRADIENT (z-space, matching HRLA)
# ============================================================
def profit_and_grad(z):
    x = SCALE * z
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    total_p = 0.0
    grad_z  = np.zeros(10, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T       = E_day + E_night
        w_s     = w_data[s]
        R_s     = R_data[s]
        g_s     = g_levels[s]
        B_reg_s = B_reg_data[s]

        bills   = np.array([f1 + p1*T, f2 + p2*T,
                            f3 + l3d*E_day + l3n*E_night,
                            f4 + l4d*E_day + l4n*E_night], dtype=float)
        costs   = cost_per_contract(E_day, E_night)
        margins = bills - costs

        u = -beta * np.concatenate(([0.0], (bills - delta * g_s * B_reg_s) - R_s))
        u -= np.max(u)
        P  = np.exp(u); P /= P.sum()

        total_p += w_s * np.dot(margins, P[1:])

        def dp_dir(db):
            db = np.asarray(db, dtype=float)
            du = -beta * np.concatenate(([0.0], db))
            dP = P * (du - np.dot(P, du))
            return np.dot(db, P[1:]) + np.dot(margins, dP[1:])

        dirs = [
            [1, 0, 0, 0], [T, 0, 0, 0],
            [0, 1, 0, 0], [0, T, 0, 0],
            [0, 0, 1, 0], [0, 0, E_day, 0], [0, 0, E_night, 0],
            [0, 0, 0, 1], [0, 0, 0, E_day], [0, 0, 0, E_night],
        ]
        for i, d in enumerate(dirs):
            grad_z[i] += w_s * dp_dir(d)

    grad_z *= SCALE
    return total_p, grad_z

# ============================================================
# SGD WITH MOMENTUM
# ============================================================
def run_sgd(M=10, K=10000, lr_init=1e-5, gamma=0.95):
    all_best_profits = []
    start_t = time.perf_counter()

    for m in range(M):
        z        = np.random.uniform(0.5, 1.5, 10)
        velocity = np.zeros(10, dtype=float)
        best_p   = -np.inf

        for k in range(1, K + 1):
            lr = lr_init * (1.0 - k / K)  # Linear decay
            p, g = profit_and_grad(z)

            if p > best_p:
                best_p = p

            velocity = gamma * velocity + lr * g
            z       += velocity
            z        = np.clip(z, 0.0, 2.5)

        all_best_profits.append(best_p)

    total_time = time.perf_counter() - start_t

    print("\n" + "=" * 50)
    print("SGD WITH MOMENTUM RESULTS")
    print("-" * 50)
    print(f"Mean Profit (€) : {np.mean(all_best_profits):.8f}")
    print(f"Std Deviation   : {np.std(all_best_profits):.8e}")
    print(f"Best Profit (€) : {np.max(all_best_profits):.8f}")
    print(f"Total Runtime   : {total_time:.4f} s")
    print(f"Mean Runtime/Run: {total_time/M:.4f} s")
    print("=" * 50)

    return all_best_profits

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_sgd()