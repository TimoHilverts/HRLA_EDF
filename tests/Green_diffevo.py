import numpy as np
import time
from scipy.optimize import differential_evolution

# ============================================================
# EXTENSION: Retailer Green Options with Static Single Competitor
# Differential Evolution version
#
# Retailer menu (N=4):
#   1) Normal Flat : (f1, p1)
#   2) Green Flat  : (f2, p2)
#   3) Normal ToU  : (f3, l3_day, l3_night)
#   4) Green ToU   : (f4, l4_day, l4_night)
#
# Competitor: ONLY Competitor A (Flat + ToU) + Fixed Fee
# Disutility DU_n = (bill_n - delta_n * g_s * B_reg_s) - R_s
# Consumption shifting: 12%
#
# Optimization variables:
# x = (f1,p1, f2,p2, f3,l3d,l3n, f4,l4d,l4n)
# ============================================================

# ============================================================
# DATA
# ============================================================

S = 10
beta = 0.03

# --------------------------
# Consumption (kWh) + 12% shifting
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

shift = 0.12
Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

# --------------------------
# Segment weights
# --------------------------
w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161, 0.116, 0.045, 0.171, 0.112, 0.101], dtype=float)
w_data = w_data / np.sum(w_data)

# ============================================================
# STATIC SINGLE COMPETITOR (A only)
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

competitor_np = np.array([
    [flat_A, flat_A],    # 0: A flat
    [tou_A[0], tou_A[1]] # 1: A ToU
], dtype=float)

fixed_fees_comp = np.array([F_A, F_A], dtype=float)

# Reservation bill R_s = minimum total competitor bill
Es_np = np.array(Es_used, dtype=float)
comp_var_bills = Es_np @ competitor_np.T
comp_bills = comp_var_bills + fixed_fees_comp[None, :]
R_data = comp_bills.min(axis=1)

# Benchmark bill B_reg for green disutility credit
lambda_reg = np.mean(competitor_np, axis=0)
f_reg = float(np.mean(fixed_fees_comp))
B_reg_data = f_reg + (Es_np @ lambda_reg.reshape(2, 1)).flatten()

# ============================================================
# GREEN COMPONENT
# ============================================================
g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00, 0.02, 0.02, 0.00, 0.00, 0.04], dtype=float)

# Retailer contracts 2 and 4 are green
delta = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

# ============================================================
# STABLE SOFTMAX
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# PROFIT FUNCTION
# ============================================================
def profit(x: np.ndarray) -> float:
    """
    x = (f1,p1, f2,p2, f3,l3d,l3n, f4,l4d,l4n)
    Returns expected retailer revenue.
    """
    x = np.asarray(x, dtype=float).flatten()
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

    total_revenue = 0.0

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

        # Green adjustment on disutility
        green_adjust = delta * g_s * B_reg_s

        # Outside option disutility is 0
        DU_contracts = (bills - green_adjust) - R_s
        DU = np.concatenate(([0.0], DU_contracts))

        # Logit probabilities over {outside, 4 retailer contracts}
        u = -beta * DU
        P = softmax_stable(u)

        # Retailer revenue excludes outside option
        seg_revenue = np.dot(bills, P[1:])
        total_revenue += w_s * seg_revenue

    return float(total_revenue)

# ============================================================
# OBJECTIVE (for minimization)
# ============================================================
def objective_x(x: np.ndarray) -> float:
    return -profit(x)

# ============================================================
# SCALING (same spirit as HRLA version)
# ============================================================
SCALE = np.array([
    F_A, flat_A,
    F_A, flat_A,
    F_A, tou_A[0], tou_A[1],
    F_A, tou_A[0], tou_A[1]
], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def objective_z(z: np.ndarray) -> float:
    return objective_x(to_x(z))

# ============================================================
# OPTIONAL: helper to inspect choice probabilities
# ============================================================
def compute_choice_table(x: np.ndarray) -> np.ndarray:
    """
    Returns an S x 5 table with probabilities for:
    [outside, R1 normal flat, R2 green flat, R3 normal ToU, R4 green ToU]
    """
    x = np.asarray(x, dtype=float).flatten()
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

    table = np.zeros((S, 5), dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night

        R_s = R_data[s]
        g_s = g_levels[s]
        B_reg_s = B_reg_data[s]

        b1 = f1 + p1 * T
        b2 = f2 + p2 * T
        b3 = f3 + l3d * E_day + l3n * E_night
        b4 = f4 + l4d * E_day + l4n * E_night
        bills = np.array([b1, b2, b3, b4], dtype=float)

        green_adjust = delta * g_s * B_reg_s
        DU_contracts = (bills - green_adjust) - R_s
        DU = np.concatenate(([0.0], DU_contracts))

        u = -beta * DU
        P = softmax_stable(u)
        table[s, :] = P

    return table

# ============================================================
# BOUNDS IN SCALED SPACE
# ============================================================
# These are example bounds; adjust if you want tighter economics-driven ranges.
#
# Ordering in z-space:
# z = (f1/F_A, p1/flat_A,
#      f2/F_A, p2/flat_A,
#      f3/F_A, l3d/tou_A_day, l3n/tou_A_night,
#      f4/F_A, l4d/tou_A_day, l4n/tou_A_night)
#
bounds = [
    (0.0, 3.0),   # f1
    (0.5, 2.0),   # p1
    (0.0, 2.0),   # f2
    (0.5, 2.0),   # p2
    (0.0, 2.0),   # f3
    (0.5, 2.0),   # l3_day
    (0.5, 2.5),   # l3_night
    (0.0, 2.0),   # f4
    (0.5, 2.0),   # l4_day
    (0.5, 2.5),   # l4_night
]

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("\nRunning Differential Evolution on fully extended model...\n")

    start = time.perf_counter()

    result = differential_evolution(
        objective_z,
        bounds=bounds,
        strategy="best1bin",
        maxiter=2000,
        popsize=25,
        tol=1e-7,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        seed=0,
        disp=True,
        workers=1,          # set to -1 if you want parallel workers
        updating="deferred" # useful when using parallel workers
    )

    end = time.perf_counter()
    runtime = end - start

    # Extract best solution
    z_best = result.x
    x_best = to_x(z_best)
    best_profit = profit(x_best)

    # Unpack for reporting
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = x_best

    print("\n========== DIFFERENTIAL EVOLUTION RESULT ==========\n")
    print("Success                 :", result.success)
    print("Message                 :", result.message)

    print("\nBest z (scaled)         :", z_best)
    print("Best x (tariffs)        :", x_best)

    print("\nRetailer contracts:")
    print("R1 Normal Flat          : f1 =", f1, ", p1 =", p1)
    print("R2 Green Flat           : f2 =", f2, ", p2 =", p2)
    print("R3 Normal ToU           : f3 =", f3, ", l3_day =", l3d, ", l3_night =", l3n)
    print("R4 Green ToU            : f4 =", f4, ", l4_day =", l4d, ", l4_night =", l4n)

    print("\nBest revenue (€)        :", best_profit)

    print("\nFunction evaluations    :", result.nfev)
    print("Iterations              :", result.nit)
    print("Runtime (seconds)       :", runtime)

    # Optional: print segment-wise choice probabilities
    choice_table = compute_choice_table(x_best)
    print("\nChoice probabilities per segment [Outside, R1, R2, R3, R4]:\n")
    np.set_printoptions(precision=6, suppress=True)
    print(choice_table)

    print("\n===================================================\n")