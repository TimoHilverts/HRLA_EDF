import numpy as np
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor
import time # Ensure this is imported at the top


# ============================================================
# EXTENSION: Retailer Green Options - Explicit Component Costs
# Integrated with HRLA Optimizer
# ============================================================

# ============================================================
# DATA & PARAMETERS
# ============================================================
S     = 10
beta  = 0.02
shift = 0.12

# ============================================================
# EXPLICIT COST STRUCTURE (Euros)
# ============================================================
C_DAY   = 0.29    # Total variable cost for Day energy
C_NIGHT = 0.21    # Total variable cost for Night energy
C_GREEN = 0.010   # REC / Green procurement premium per kWh
C_SERVE = 110.0   # Annual fixed cost (admin, metering, billing)

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

# Derived Cost for Flat contracts (weighted average based on population)
def get_avg_cost():
    total_day = np.sum(Es_original[:, 0])
    total_night = np.sum(Es_original[:, 1])
    return (total_day * C_DAY + total_night * C_NIGHT) / (total_day + total_night)

C_FLAT = get_avg_cost()

# ============================================================
# COMPETITOR & RESERVATION BILL
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

competitor_np   = np.array([[flat_A, flat_A],
                             [tou_A[0], tou_A[1]]], dtype=float)
fixed_fees_comp = np.array([F_A, F_A], dtype=float)

comp_bills = (Es_used @ competitor_np.T) + fixed_fees_comp[None, :]
R_data     = comp_bills.min(axis=1)

# Benchmark bill for green utility credit
lambda_reg = np.mean(competitor_np, axis=0)
B_reg_data = F_A + (Es_used @ lambda_reg)

# Green preference levels
g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00,
                     0.02, 0.02, 0.00, 0.00, 0.04], dtype=float)
delta    = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)

# ============================================================
# COST FUNCTION
# ============================================================
def cost_per_contract(E_day: float, E_night: float) -> np.ndarray:
    """Returns cost vector [K1, K2, K3, K4] based on explicit components."""
    T = E_day + E_night
    c_tou_base = C_SERVE + (C_DAY * E_day + C_NIGHT * E_night)
    c_flat_base = C_SERVE + (C_FLAT * T)
    
    return np.array([
        c_flat_base,                        # R1 Normal Flat
        c_flat_base + (C_GREEN * T),        # R2 Green Flat
        c_tou_base,                         # R3 Normal ToU
        c_tou_base + (C_GREEN * T),         # R4 Green ToU
    ], dtype=float)

# ============================================================
# SCALING & BOUNDS
# ============================================================
SCALE = np.array([
    F_A, flat_A, F_A, flat_A,
    F_A, tou_A[0], tou_A[1],
    F_A, tou_A[0], tou_A[1]
], dtype=float)

BOUNDS = [(0.0, 3.0)] * 10

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

# ============================================================
# SOFTMAX
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# PROFIT + ANALYTIC GRADIENT
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float).flatten()
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

    total_profit = 0.0
    grad_profit  = np.zeros(10, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s     = w_data[s]
        R_s     = R_data[s]
        g_s     = g_levels[s]
        B_reg_s = B_reg_data[s]

        # Bills
        b1 = f1 + p1 * T
        b2 = f2 + p2 * T
        b3 = f3 + l3d * E_day + l3n * E_night
        b4 = f4 + l4d * E_day + l4n * E_night
        bills = np.array([b1, b2, b3, b4], dtype=float)

        # Costs and margins
        costs   = cost_per_contract(E_day, E_night)
        margins = bills - costs

        # Disutilities
        green_adjust = delta * g_s * B_reg_s
        DU_contracts = (bills - green_adjust) - R_s
        DU = np.concatenate(([0.0], DU_contracts))

        u = -beta * DU
        P = softmax_stable(u)

        # Expected profit
        total_profit += w_s * float(np.dot(margins, P[1:]))

        def dp_direction(dbills: np.ndarray) -> float:
            dbills = np.asarray(dbills, dtype=float).reshape(4,)
            dDU    = np.concatenate(([0.0], dbills))
            du     = -beta * dDU
            du_bar = float(np.dot(P, du))
            dP     = P * (du - du_bar)
            # Derivative of profit = Prob * dBills + Margins * dProb
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

# ============================================================
# GRADIENT CHECKER
# ============================================================
def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g

def initial() -> np.ndarray:
    return np.random.uniform(low=0.0, high=2.0, size=10)

# ============================================================
# MAIN EXECUTION
# ============================================================
title = "ExplicitCosts_Green_HRLA"
d     = 10

if __name__ == "__main__":
    print("Explicit Cost Structure:")
    print(f"  Day Energy    : {C_DAY:.4f} €/kWh")
    print(f"  Night Energy  : {C_NIGHT:.4f} €/kWh")
    print(f"  Flat Base (avg): {C_FLAT:.4f} €/kWh")
    print(f"  Green Premium : {C_GREEN:.4f} €/kWh")
    print(f"  Fixed Serve   : {C_SERVE:.2f} €/year")
    print()

    # Gradient check
    x_test     = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd       = finite_diff_grad(U_x, x_test, eps=1e-6)
    print("Gradient check (Max abs error):", float(np.max(np.abs(g_analytic - g_fd))))
    print()

    algorithm = GO.HRLA(
        d=d, M=10, N=1, K=10000, h=1e-4,
        a_min=0.1,
        title=title, U=U_z, dU=dU_z, initial=initial,
    )

    # 2. Generate Samples with Runtime Tracking
    print(f"Starting HRLA Optimization (M=10, N=1, K=20000)...")
    start_time = time.perf_counter()
    
    samples_filename = algorithm.generate_samples(
        As=[100000], sim_annealing=True
    )
    
    end_time = time.perf_counter()
    total_runtime = end_time - start_time

    # 3. Post-Processing
    print("\n" + "="*30)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*30)
    print(f"Total Runtime for M=10 runs: {total_runtime:.4f} seconds")
    print(f"Average Runtime per run:    {total_runtime/10:.4f} seconds")
    print("="*30 + "\n")

    postprocessor = PostProcessor(samples_filename)
    measured = [10000]
    
    # These will print your LaTeX-ready tables for Mean, Std, and Best
    postprocessor.compute_tables(measured, 1, "mean")
    postprocessor.compute_tables(measured, 1, "std")
    postprocessor.compute_tables(measured, 1, "best")