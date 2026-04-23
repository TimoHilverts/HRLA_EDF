import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# ============================================================
# DATA & CONSTANTS
# ============================================================
S = 10
shift = 0.12
ALPHA = 0.92
ALPHA_GREEN = 0.05
C_FIX = 0

Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
    [4896.07, 401.88],
], dtype=float)

def shifting_cons(E, s):
    return np.array([(1.0 - s) * E[0], E[1] + s * E[0]], dtype=float)

Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)
w_data = np.array([0.125, 0.057, 0.068, 0.044, 0.161, 0.116, 0.045, 0.171, 0.112, 0.101])
w_data /= np.sum(w_data)

tou_A, flat_A, F_A = np.array([0.355, 0.250]), 0.342, 196.0
competitor_np = np.array([[flat_A, flat_A], [tou_A[0], tou_A[1]]])
fixed_fees_comp = np.array([F_A, F_A])
comp_bills = (Es_used @ competitor_np.T) + fixed_fees_comp[None, :]
R_data = comp_bills.min(axis=1)

lambda_reg = np.mean(competitor_np, axis=0)
B_reg_data = F_A + (Es_used @ lambda_reg)
g_levels = np.array([0.00, 0.04, 0.00, 0.04, 0.00, 0.02, 0.02, 0.00, 0.00, 0.04])
delta = np.array([0.0, 1.0, 0.0, 1.0])

# Variable Cost Floors
c_flat = ALPHA * flat_A
c_flat_green = (ALPHA + ALPHA_GREEN) * flat_A
c_day = ALPHA * tou_A[0]
c_night = ALPHA * tou_A[1]
c_day_green = (ALPHA + ALPHA_GREEN) * tou_A[0]
c_night_green = (ALPHA + ALPHA_GREEN) * tou_A[1]

SCALE = np.array([F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]])

# ============================================================
# FUNCTIONS
# ============================================================
def cost_per_contract(E_day, E_night):
    T = E_day + E_night
    return np.array([
        C_FIX + c_flat * T,
        C_FIX + c_flat_green * T,
        C_FIX + c_day * E_day + c_night * E_night,
        C_FIX + c_day_green * E_day + c_night_green * E_night,
    ])

def profit(x, current_beta):
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = x
    total_profit = 0.0
    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        bills = np.array([f1+p1*T, f2+p2*T, f3+l3d*E_day+l3n*E_night, f4+l4d*E_day+l4n*E_night])
        costs = cost_per_contract(E_day, E_night)
        
        DU = (bills - delta * g_levels[s] * B_reg_data[s]) - R_data[s]
        u = -current_beta * np.concatenate(([0.0], DU))
        
        # Softmax stable
        u_max = np.max(u)
        e = np.exp(u - u_max)
        P = e / e.sum()
        
        margins = bills - costs
        total_profit += w_data[s] * np.dot(margins, P[1:])
    return total_profit

# ============================================================
# SWEEP EXECUTION
# ============================================================
if __name__ == "__main__":
    betas = np.array([1e-4, 3e-4, 7e-4, 9e-4, 1e-3, 1.2e-3, 2e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0])
    max_profits = []
    
    bounds = [(0, 2.5)] * 10 

    print(f"{'Beta':>10} | {'Expected Profit (€)':>20} | {'Time (s)':>10}")
    print("-" * 50)

    for b in betas:
        start_t = time.perf_counter()
        
        # Best1bin is robust for this type of landscape
        res = differential_evolution(
            lambda z: -profit(SCALE * z, b),
            bounds=bounds,
            strategy='best1bin',
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=True
        )
        
        max_profits.append(-res.fun)
        elapsed = time.perf_counter() - start_t
        print(f"{b:10.5f} | {-res.fun:20.2f} | {elapsed:10.2f}")

    # ============================================================
    # LOG-LINEAR PLOTTING
    # ============================================================
    plt.figure(figsize=(10, 6))
    plt.plot(betas, max_profits, marker='o', color='navy', linewidth=2)
    
    # Using log scale for X because the values span 5 orders of magnitude
    plt.xscale('log')
    
    plt.title("Expected Retailer Profit vs. Price Sensitivity (Log Scale)", fontsize=14)
    plt.xlabel("Beta (Consumer Sensitivity) - Log Scale", fontsize=12)
    plt.ylabel("Optimal Expected Annual Profit (€)", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.show()