import numpy as np
import time
from scipy.optimize import differential_evolution

# ============================================================
# EXTENSION: Retailer Green Options - Explicit Component Costs
# Differential Evolution version
# ============================================================

# ============================================================
# DATA & PARAMETERS
# ============================================================
S    = 10
beta = 0.02
shift = 0.12

# ============================================================
# EXPLICIT COST STRUCTURE (Euros)
# ============================================================
# Procurement + Grid + Taxes (The "Thin Margin" Profile)
C_DAY   = 0.29    # Total variable cost for Day energy
C_NIGHT = 0.21   # Total variable cost for Night energy
C_GREEN = 0.010   # REC / Green procurement premium per kWh
C_SERVE = 110.0   # Annual fixed cost (admin, metering, billing)

# Derived Cost for Flat contracts (weighted average of day/night)
# We calculate a reference flat cost based on the total population consumption
Es_initial_totals = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

# Calculate a population-weighted average cost per kWh for Flat contracts
def get_avg_cost():
    total_day = np.sum(Es_initial_totals[:, 0])
    total_night = np.sum(Es_initial_totals[:, 1])
    return (total_day * C_DAY + total_night * C_NIGHT) / (total_day + total_night)

C_FLAT = get_avg_cost()

# ============================================================
# CONSUMPTION DATA
# ============================================================
def shifting_cons(E: np.ndarray, s: float) -> np.ndarray:
    return np.array([(1.0 - s) * E[0], E[1] + s * E[0]], dtype=float)

Es_used = np.array([shifting_cons(E, shift) for E in Es_initial_totals], dtype=float)

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
# COST FUNCTION
# ============================================================
def cost_per_contract(E_day, E_night):
    """Returns cost vector for R1-R4 based on explicit components."""
    T = E_day + E_night
    # Base procurement for ToU
    c_tou_base = C_SERVE + (C_DAY * E_day + C_NIGHT * E_night)
    # Base procurement for Flat (uses the derived average)
    c_flat_base = C_SERVE + (C_FLAT * T)
    
    return np.array([
        c_flat_base,                      # R1 Normal Flat
        c_flat_base + (C_GREEN * T),      # R2 Green Flat
        c_tou_base,                       # R3 Normal ToU
        c_tou_base + (C_GREEN * T),       # R4 Green ToU
    ], dtype=float)

# ============================================================
# OPTIMIZATION CORE
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    u = u - np.max(u)
    e = np.exp(u)
    return e / e.sum()

def profit(x: np.ndarray) -> float:
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    total_profit = 0.0

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        
        # Bills
        bills = np.array([
            f1 + p1 * T,
            f2 + p2 * T,
            f3 + l3d * E_day + l3n * E_night,
            f4 + l4d * E_day + l4n * E_night,
        ], dtype=float)

        costs = cost_per_contract(E_day, E_night)

        # Choice Model
        DU_contracts = (bills - delta * g_levels[s] * B_reg_data[s]) - R_data[s]
        u = -beta * np.concatenate(([0.0], DU_contracts))
        P = softmax_stable(u)

        # Profit = Prob * (Revenue - Cost)
        margins = bills - costs
        total_profit += w_data[s] * float(np.dot(margins, P[1:]))

    return total_profit

def objective_z(z: np.ndarray) -> float:
    return -profit(SCALE * z)

# ============================================================
# SCALING & BOUNDS
# ============================================================
SCALE = np.array([F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]], dtype=float)

# Define bounds in scaled space [0, 2.0]
# 0.0 is zero price, 2.0 is 200% of competitor price.
bounds = [(0, 2.0) for _ in range(10)]

def compute_choice_table(x: np.ndarray) -> np.ndarray:
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)
    table = np.zeros((S, 5), dtype=float)
    for s in range(S):
        E_day, E_night = Es_used[s]
        T, R_s, g_s, B_s = E_day + E_night, R_data[s], g_levels[s], B_reg_data[s]
        bills = np.array([f1 + p1*T, f2 + p2*T, f3 + l3d*E_day + l3n*E_night, f4 + l4d*E_day + l4n*E_night])
        DU = np.concatenate(([0.0], (bills - delta*g_s*B_s) - R_s))
        table[s] = softmax_stable(-beta * DU)
    return table

# ============================================================
# MAIN EXECUTION WITH HIGH-PRECISION STATISTICS
# ============================================================
if __name__ == "__main__":
    N_RUNS = 10  # Number of times to run the optimizer for stats
    all_profits = []
    all_runtimes = []
    best_overall_result = None
    max_profit_found = -np.inf

    print(f"Starting Benchmark: {N_RUNS} Replications of Differential Evolution...")
    print(f"Population size: 1, Max Iterations: 1500\n")

    global_start = time.perf_counter()

    for i in range(N_RUNS):
        start_iter = time.perf_counter()
        
        # Vary seed for statistical distribution
        result = differential_evolution(
            objective_z,
            bounds=bounds,
            strategy="best1bin",
            maxiter=1500,
            popsize=1,
            tol=1e-7,
            polish=True,
            seed=i, 
            disp=False
        )
        
        end_iter = time.perf_counter()
        
        current_profit = -result.fun
        all_profits.append(current_profit)
        all_runtimes.append(end_iter - start_iter)
        
        if current_profit > max_profit_found:
            max_profit_found = current_profit
            best_overall_result = result
            
        print(f"  Run {i+1}/{N_RUNS}: Profit = {current_profit:.8f} € | Time = {end_iter - start_iter:.8f}s")

    global_end = time.perf_counter()

    # Calculate Statistics rounded to 8 decimals
    mean_profit = round(float(np.mean(all_profits)), 8)
    std_profit  = round(float(np.std(all_profits)), 8)
    best_profit = round(float(max_profit_found), 8)
    
    avg_runtime   = round(float(np.mean(all_runtimes)), 8)
    total_runtime = round(float(global_end - global_start), 8)

    # Extract best parameters
    x_best = SCALE * best_overall_result.x
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = x_best

    # ============================================================
    # FINAL REPORTING (8 Decimal Precision)
    # ============================================================
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY (Over {} runs)".format(N_RUNS))
    print("="*60)
    # Using :.8f string formatting to ensure 8 decimals are displayed
    print(f"Mean Profit         : {mean_profit:.8f} €")
    print(f"Std. Dev. of Profit : {std_profit:.8f} €")
    print(f"Best Profit Found   : {best_profit:.8f} €")
    print("-"*60)
    print(f"Total Runtime       : {total_runtime:.8f} s")
    print(f"Avg. Runtime/Run    : {avg_runtime:.8f} s")
    print("="*60)

    print("\n========== BEST FOUND CONTRACTS ==========\n")
    print(f"R1 Normal Flat : f1 = {f1:.8f} €/yr,  p1 = {p1:.8f} €/kWh")
    print(f"R2 Green Flat  : f2 = {f2:.8f} €/yr,  p2 = {p2:.8f} €/kWh")
    print(f"R3 Normal ToU  : f3 = {f3:.8f} €/yr,  day = {l3d:.8f},  night = {l3n:.8f} €/kWh")
    print(f"R4 Green ToU   : f4 = {f4:.8f} €/yr,  day = {l4d:.8f},  night = {l4n:.8f} €/kWh")

    print("\nChoice probabilities for the BEST result [Outside, R1, R2, R3, R4]:")
    np.set_printoptions(precision=8, suppress=True)
    print(compute_choice_table(x_best))