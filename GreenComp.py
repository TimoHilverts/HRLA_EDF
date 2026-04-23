import numpy as np
import sympy as sp
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# FULL MODEL:
# - weights
# - competitors: 2 suppliers x {flat, ToU} + annual fixed fees
# - reservation bill = min competitor total bill
# - retailer fixed fees
# - consumption shifting
# - green component
# Menu (N=4):
#   1) normal flat + fixed fee
#   2) green flat + fixed fee
#   3) normal ToU + fixed fee
#   4) green ToU + fixed fee

N_contracts = 4
S = 10
D = 2

beta = 0.02

# Consumption + shifting

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

def shifting_cons(E, shift):
    E_day, E_night = E
    E_day_new = (1 - shift) * E_day
    E_night_new = E_night + shift * E_day
    return np.array([E_day_new, E_night_new])

shift = 0.15
Es_used = np.array([shifting_cons(E, shift) for E in Es_original], dtype=float)

# Segment weights

w_data = np.array([
    0.125, 0.057, 0.068, 0.044, 0.161,
    0.116, 0.045, 0.171, 0.112, 0.101
], dtype=float)
w_data = w_data / np.sum(w_data)

# Competitors: 2 suppliers x {flat, ToU} + annual fixed fees

tou_A  = np.array([0.37, 0.28], dtype=float)
flat_A = np.array([0.345, 0.345], dtype=float)
F_A    = 160.0

tou_B  = np.array([0.35, 0.30], dtype=float)
flat_B = np.array([0.330, 0.330], dtype=float)
F_B    = 152.0

competitors = [flat_A, tou_A, flat_B, tou_B]
fixed_fees_comp_list = [F_A, F_A, F_B, F_B]

competitors_np = np.array(competitors, dtype=float)     # (C,2)
Es_np = np.array(Es_used, dtype=float)                  # (S,2)

comp_var_bills = Es_np @ competitors_np.T                # (S,C)
comp_fixed = np.array(fixed_fees_comp_list, dtype=float) # (C,)
comp_bills = comp_var_bills + comp_fixed[None, :]        # (S,C)

R_data = comp_bills.min(axis=1)                          # (S,)

# Green component

g_levels = np.array([0.04, 0.02, 0.00, 0.02, 0.01, 0.03, 0.00, 0.02, 0.01, 0.04], dtype=float)
delta = np.array([0, 1, 0, 1], dtype=float)  # contracts 2 and 4 are green

# Regulated tariff and fixed fee: average over competitor MENU
lambda_reg = np.mean(np.array(competitors, dtype=float), axis=0)  # (2,)
f_reg = float(np.mean(np.array(fixed_fees_comp_list, dtype=float)))

B_reg_data = f_reg + (Es_np @ lambda_reg.reshape(2, 1)).flatten()  # (S,)

# Decision variables (10D)
# (f1,p1,f2,p2,f3,l3d,l3n,f4,l4d,l4n)

f1, p1 = sp.symbols("f1 p1", real=True)
f2, p2 = sp.symbols("f2 p2", real=True)
f3, l3_day, l3_night = sp.symbols("f3 l3_day l3_night", real=True)
f4, l4_day, l4_night = sp.symbols("f4 l4_day l4_night", real=True)

variables = (f1, p1, f2, p2, f3, l3_day, l3_night, f4, l4_day, l4_night)

fixed_fees = [f1, f2, f3, f4]

lambda1 = sp.Matrix([p1, p1])
lambda2 = sp.Matrix([p2, p2])
lambda3 = sp.Matrix([l3_day, l3_night])
lambda4 = sp.Matrix([l4_day, l4_night])
lambdas = [lambda1, lambda2, lambda3, lambda4]

def symbolic_profit(beta_val: float, w: np.ndarray) -> sp.Expr:
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es_used[s])

        R_s = sp.Float(R_data[s])
        B_reg_s = sp.Float(B_reg_data[s])
        g_s = sp.Float(g_levels[s])

        bills = [
            fixed_fees[n] + E_s.dot(lambdas[n])
            for n in range(N_contracts)
        ]

        DU_outside = sp.Integer(0)
        DUs = [DU_outside]
        for n in range(N_contracts):
            delta_n = sp.Float(delta[n])
            DU_n = (bills[n] - delta_n * g_s * B_reg_s) - R_s
            DUs.append(DU_n)

        exp_terms = [sp.exp(-beta_val * u) for u in DUs]
        denom = sum(exp_terms)
        probs = [e / denom for e in exp_terms]  # probs[0]=outside, probs[1..4]=contracts

        total += w[s] * sum(bills[n] * probs[n + 1] for n in range(N_contracts))

    return total

profit_sym = symbolic_profit(beta, w_data)
U_sym = -profit_sym
gradU_sym = [sp.diff(U_sym, v) for v in variables]

U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

def U_x(x):
    return float(U_func(*x))

def dU_x(x):
    return np.array(gradU_func(*x), dtype=float).flatten()

# Scaling (10D)

# Fixed fee scale: average annual competitor fixed fee
fixed_fee_scale = float(np.mean(np.array([F_A, F_B], dtype=float)))

# Flat price scale: mean of competitor flat tariffs (flat tariffs are identical day/night)
flat_scale = float(np.mean(np.array([flat_A[0], flat_B[0]], dtype=float)))

# ToU scales: mean day and mean night across ToU competitors only
day_scale   = float(np.mean(np.array([tou_A[0], tou_B[0]], dtype=float)))
night_scale = float(np.mean(np.array([tou_A[1], tou_B[1]], dtype=float)))

SCALE = np.array([
    fixed_fee_scale, flat_scale,          # (f1, p1)  normal flat
    fixed_fee_scale, flat_scale,          # (f2, p2)  green flat
    fixed_fee_scale, day_scale, night_scale,  # (f3, l3_day, l3_night) normal ToU
    fixed_fee_scale, day_scale, night_scale   # (f4, l4_day, l4_night) green ToU
], dtype=float)

def to_x(z):
    return SCALE * np.asarray(z, dtype=float)

def U_z(z):
    return U_x(to_x(z))

def dU_z(z):
    return SCALE * dU_x(to_x(z))

# Run

title = "GreenComp_Full"
d = len(variables)

def initial():
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(loc=0.0, scale=0.05, size=d)
    return z0

if __name__ == "__main__":
    print("Starting Optimization (N=2; 1 competitor; bill-based logit; UNIFORM weights; scaled variables)")

    algorithm = GO.HRLA(
        d=d,
        M=1,
        N=1,          # number of parallel trajectories (keep 1 unless you want more)
        K=2000,
        h=0.0001,
        title=title,
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=[20],
        #As=[5, 50, 100, 500, 1000, 3000, 5000],
        sim_annealing=True,
    )

    print(f"Optimization finished. Samples saved to: {samples_filename}")
    print("NOTE: saved samples are z-vectors. Convert to real tariffs via x = to_x(z).")

    #postprocessing (tables + best trajectories) ---
    postprocessor = PostProcessor(samples_filename)

    #measured = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    measured = [1, 3, 5, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    postprocessor.compute_tables(measured, 1, "best")
    bests = postprocessor.get_best(measured=measured, dpi=1)
