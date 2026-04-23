import numpy as np
import time
from scipy.optimize import differential_evolution

# ============================================================
# EXTENSION: Retailer Green Options with Static Single Competitor
# Differential Evolution version — Economically grounded cost model
#
# Retailer menu (N=4):
#   1) Normal Flat : (f1, p1)
#   2) Green Flat  : (f2, p2)
#   3) Normal ToU  : (f3, l3_day, l3_night)
#   4) Green ToU   : (f4, l4_day, l4_night)
#
# Cost structure:
#   Variable costs are set as a fraction alpha of competitor marginal
#   rates, reflecting that competitor prices are full retail prices
#   including wholesale, network charges, taxes and margin. This avoids
#   the need to decompose costs into components without regulatory data.
#   Green contracts carry a small additional variable cost premium
#   reflecting renewable energy certificate costs.
#   Fixed costs reflect customer management, billing and overhead.
#
#   Reference for cost-to-revenue approach:
#   ACM Monitor Energiemarkt (Netherlands Authority for Consumers
#   and Markets), annual retail market monitor reports.
# ============================================================

# ============================================================
# DATA & PARAMETERS
# ============================================================
S    = 10
beta = 0.02
shift = 0.12

# ============================================================
# COST STRUCTURE
# ============================================================
# alpha: fraction of competitor marginal rate representing variable cost
# Implies (1-alpha) net margin on variable component
# Set to 0.92 => 8% net margin, consistent with competitive retail markets
ALPHA       = 0.92

# Green premium: additional fraction of competitor rate for renewable
# energy certificate procurement. Typical EKO certificate cost in
# Netherlands: 0.01-0.03 euros/kWh. As fraction of flat_A=0.342:
# 0.02/0.342 ~ 0.058. We use 0.03 to be conservative.
ALPHA_GREEN = 0.03

# Fixed cost per customer per year (euros)
# Covers billing, customer service, metering data, compliance overhead
# Consistent with European retail electricity market estimates of 50-100 euros
C_FIX = 80

# ============================================================
# CONSUMPTION DATA
# ============================================================
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

# ============================================================
# COMPETITOR & RESERVATION BILL
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

competitor_np    = np.array([[flat_A, flat_A],
                              [tou_A[0], tou_A[1]]], dtype=float)
fixed_fees_comp  = np.array([F_A, F_A], dtype=float)

comp_bills = (Es_used @ competitor_np.T) + fixed_fees_comp[None, :]
R_data     = comp_bills.min(axis=1)

# Benchmark bill for green utility credit
lambda_reg  = np.mean(competitor_np, axis=0)
B_reg_data  = F_A + (Es_used @ lambda_reg)

# Green preference levels per segment
g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00,
                     0.02, 0.02, 0.00, 0.00, 0.04], dtype=float)
delta    = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

# ============================================================
# DERIVED VARIABLE COSTS
# Variable cost per kWh for each contract type, derived from
# competitor prices scaled by alpha.
#
# For flat contracts: single blended rate
#   c_flat        = alpha * flat_A
#   c_flat_green  = (alpha + alpha_green) * flat_A
#
# For ToU contracts: separate day/night rates
#   c_day         = alpha * tou_A[0]
#   c_night       = alpha * tou_A[1]
#   c_day_green   = (alpha + alpha_green) * tou_A[0]
#   c_night_green = (alpha + alpha_green) * tou_A[1]
# ============================================================
c_flat        = ALPHA       * flat_A          # 0.92 * 0.342 = 0.3146
c_flat_green  = (ALPHA + ALPHA_GREEN) * flat_A  # 0.95 * 0.342 = 0.3249
c_day         = ALPHA       * tou_A[0]        # 0.92 * 0.355 = 0.3266
c_night       = ALPHA       * tou_A[1]        # 0.92 * 0.250 = 0.2300
c_day_green   = (ALPHA + ALPHA_GREEN) * tou_A[0]  # 0.95 * 0.355 = 0.3373
c_night_green = (ALPHA + ALPHA_GREEN) * tou_A[1]  # 0.95 * 0.250 = 0.2375

print("Variable cost floors (euros/kWh):")
print(f"  R1 Normal Flat : {c_flat:.4f}")
print(f"  R2 Green Flat  : {c_flat_green:.4f}")
print(f"  R3 Normal ToU  : day={c_day:.4f}, night={c_night:.4f}")
print(f"  R4 Green ToU   : day={c_day_green:.4f}, night={c_night_green:.4f}")
print(f"  Fixed cost     : {C_FIX:.1f} euros/year\n")

# ============================================================
# COST FUNCTION
# Total annual cost of serving segment s under contract n
# ============================================================
def cost_per_contract(E_day, E_night):
    """Returns cost vector of length 4 for contracts R1-R4."""
    T = E_day + E_night
    return np.array([
        C_FIX + c_flat        * T,        # R1 Normal Flat
        C_FIX + c_flat_green  * T,        # R2 Green Flat
        C_FIX + c_day         * E_day + c_night       * E_night,  # R3 Normal ToU
        C_FIX + c_day_green   * E_day + c_night_green * E_night,  # R4 Green ToU
    ], dtype=float)

# ============================================================
# SOFTMAX
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    u = u - np.max(u)
    e = np.exp(u)
    return e / e.sum()

# ============================================================
# PROFIT FUNCTION
# ============================================================
def profit(x: np.ndarray) -> float:
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    total_profit = 0.0

    for s in range(S):
        E_day, E_night = Es_used[s]
        T    = E_day + E_night
        w_s  = w_data[s]
        R_s  = R_data[s]
        g_s  = g_levels[s]
        B_s  = B_reg_data[s]

        # Bills
        bills = np.array([
            f1 + p1 * T,
            f2 + p2 * T,
            f3 + l3d * E_day + l3n * E_night,
            f4 + l4d * E_day + l4n * E_night,
        ], dtype=float)

        # Costs
        costs = cost_per_contract(E_day, E_night)

        # Disutilities
        DU_contracts = (bills - delta * g_s * B_s) - R_s
        u = -beta * np.concatenate(([0.0], DU_contracts))
        P = softmax_stable(u)

        # Profit = sum of margin * probability
        margins = bills - costs
        total_profit += w_s * float(np.dot(margins, P[1:]))

    return total_profit

def objective_z(z: np.ndarray) -> float:
    return -profit(SCALE * z)

# ============================================================
# SCALING
# ============================================================
SCALE = np.array([
    F_A, flat_A, F_A, flat_A,
    F_A, tou_A[0], tou_A[1],
    F_A, tou_A[0], tou_A[1]
], dtype=float)

# ============================================================
# BOUNDS
# Lower bounds derived directly from cost structure:
#   - Fixed fee lower bound: C_FIX = 70 euros
#   - Marginal rate lower bounds: variable cost per contract
# Upper bounds: 1.5x competitor price (commercially implausible above this)
# All converted to scaled space.
# ============================================================
# bounds = [
#     # R1 Normal Flat
#     (C_FIX / F_A,       1.5),   # f1
#     (c_flat / flat_A,   1.5),   # p1

#     # R2 Green Flat
#     (C_FIX / F_A,            1.5),   # f2
#     (c_flat_green / flat_A,  1.5),   # p2

#     # R3 Normal ToU
#     (C_FIX / F_A,        1.5),   # f3
#     (c_day / tou_A[0],   1.5),   # l3_day
#     (c_night / tou_A[1], 1.5),   # l3_night

#     # R4 Green ToU
#     (C_FIX / F_A,             1.5),   # f4
#     (c_day_green / tou_A[0],  1.5),   # l4_day
#     (c_night_green / tou_A[1],1.5),   # l4_night
# ]
bounds = [
    # R1 Normal Flat
    (0,       2.5),   # f1
    (0,       2.5),   # p1

    # R2 Green Flat
    (0,       2.5),   # f2
    (0,       2.5),   # p2

    # R3 Normal ToU
    (0,       2.5),   # f3
    (0,       2.5),   # l3_day
    (0,       2.5),   # l3_night

    # R4 Green ToU
    (0,       2.5),   # f4
    (0,       2.5),   # l4_day
    (0,       2.5),   # l4_night
]

# ============================================================
# CHOICE PROBABILITY TABLE
# ============================================================
def compute_choice_table(x: np.ndarray) -> np.ndarray:
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    table = np.zeros((S, 5), dtype=float)
    for s in range(S):
        E_day, E_night = Es_used[s]
        T   = E_day + E_night
        R_s = R_data[s]
        g_s = g_levels[s]
        B_s = B_reg_data[s]
        bills = np.array([
            f1 + p1*T, f2 + p2*T,
            f3 + l3d*E_day + l3n*E_night,
            f4 + l4d*E_day + l4n*E_night,
        ])
        DU = np.concatenate(([0.0], (bills - delta*g_s*B_s) - R_s))
        table[s] = softmax_stable(-beta * DU)
    return table

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # Print implied cost floors in x-space for transparency
    print("Price lower bounds derived from cost structure (x-space):")
    print(f"  R1 fixed fee  >= {C_FIX:.1f} €/yr,  flat rate  >= {c_flat:.4f} €/kWh")
    print(f"  R2 fixed fee  >= {C_FIX:.1f} €/yr,  flat rate  >= {c_flat_green:.4f} €/kWh")
    print(f"  R3 fixed fee  >= {C_FIX:.1f} €/yr,  day rate   >= {c_day:.4f},  night rate >= {c_night:.4f} €/kWh")
    print(f"  R4 fixed fee  >= {C_FIX:.1f} €/yr,  day rate   >= {c_day_green:.4f},  night rate >= {c_night_green:.4f} €/kWh")
    print(f"\nImplied net margin at competitor prices: {(1-ALPHA)*100:.0f}% on variable component")
    print(f"Green premium on variable cost: {ALPHA_GREEN*100:.0f}% of competitor rate\n")

    print("Running Differential Evolution...\n")
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
        workers=1,
        updating="deferred"
    )

    end = time.perf_counter()
    x_best = SCALE * result.x
    best_profit_val = profit(x_best)

    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = x_best

    print("\n========== OPTIMAL CONTRACTS ==========\n")
    print(f"R1 Normal Flat : f1 = {f1:.2f} €/yr,  p1 = {p1:.4f} €/kWh")
    print(f"R2 Green Flat  : f2 = {f2:.2f} €/yr,  p2 = {p2:.4f} €/kWh")
    print(f"R3 Normal ToU  : f3 = {f3:.2f} €/yr,  day = {l3d:.4f},  night = {l3n:.4f} €/kWh")
    print(f"R4 Green ToU   : f4 = {f4:.2f} €/yr,  day = {l4d:.4f},  night = {l4n:.4f} €/kWh")

    print(f"\nExpected annual net profit : {best_profit_val:.2f} €")
    print(f"Runtime                    : {end-start:.2f} seconds")

    print("\nCompetitor reference:")
    print(f"  Fixed fee    : {F_A:.2f} €/yr")
    print(f"  Flat rate    : {flat_A:.3f} €/kWh")
    print(f"  ToU day rate : {tou_A[0]:.3f} €/kWh")
    print(f"  ToU night    : {tou_A[1]:.3f} €/kWh")

    # Margin analysis
    print("\nMargin analysis at optimal solution:")
    print(f"{'Seg':>4} {'Cost_R1':>9} {'Bill_R1':>9} {'Margin_R1':>11} "
          f"{'Cost_R3':>9} {'Bill_R3':>9} {'Margin_R3':>11}")
    print("-"*70)
    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        costs = cost_per_contract(E_day, E_night)
        b1 = f1 + p1*T
        b3 = f3 + l3d*E_day + l3n*E_night
        print(f"{s+1:>4} {costs[0]:>9.1f} {b1:>9.1f} {b1-costs[0]:>11.1f} "
              f"{costs[2]:>9.1f} {b3:>9.1f} {b3-costs[2]:>11.1f}")

    print("\nChoice probabilities [Outside, R1, R2, R3, R4]:")
    np.set_printoptions(precision=4, suppress=True)
    print(compute_choice_table(x_best))

    print("\n========================================\n")