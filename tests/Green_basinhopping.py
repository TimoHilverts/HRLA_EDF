import numpy as np
import time
from scipy.optimize import basinhopping, Bounds

# ============================================================
# EXTENSION: Retailer Green Options with Static Single Competitor
# - Retailer menu (N=4):
#   1) Normal Flat : (f1, p1)
#   2) Green Flat  : (f2, p2)
#   3) Normal ToU  : (f3, l3_day, l3_night)
#   4) Green ToU   : (f4, l4_day, l4_night)
# - Competitor: ONLY Competitor A (Flat + ToU) + Fixed Fee
# - Disutility DU_n = (bill_n - delta_n * g_s * B_reg_s) - R_s
# - Consumption shifting: 12% shift
# - Optimization: Basinhopping (L-BFGS-B), N_RUNS=20, NITER=200
# ============================================================

S = 10
beta = 0.01

# --------------------------
# Consumption (kWh) + 12% shifting
# --------------------------
Es_original = np.array([
    [3081.18, 267.45], [4727.04, 646.87], [2286.51, 499.64],
    [7308.55, 599.22], [1779.13, 344.83], [3740.28, 542.46],
    [2724.48, 824.06], [3383.92, 388.33], [1762.53, 224.38],
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
# STATIC SINGLE Competitor (A only)
# ============================================================
tou_A  = np.array([0.355, 0.250], dtype=float)
flat_A = 0.342
F_A    = 196.0

competitor_np = np.array([
    [flat_A, flat_A],
    [tou_A[0], tou_A[1]]
], dtype=float)

fixed_fees_comp = np.array([F_A, F_A], dtype=float)

# Reservation bill R_s (min of A-flat and A-ToU)
Es_np = np.array(Es_used, dtype=float)
comp_var_bills = Es_np @ competitor_np.T
comp_bills = comp_var_bills + fixed_fees_comp[None, :]
R_data = comp_bills.min(axis=1)

# Benchmark bill (B_reg) for green disutility credit
lambda_reg = np.mean(competitor_np, axis=0)
f_reg = float(np.mean(fixed_fees_comp))
B_reg_data = f_reg + (Es_np @ lambda_reg.reshape(2, 1)).flatten()

# ============================================================
# Green component
# ============================================================
g_levels = np.array([0.00, 0.04, 0.0, 0.04, 0.0, 0.02, 0.02, 0.00, 0.0, 0.04], dtype=float)
delta = np.array([0, 1, 0, 1], dtype=float)

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit + analytic gradient (10D x-space)
# x = (f1,p1, f2,p2, f3,l3d,l3n, f4,l4d,l4n)
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple:
    x = np.asarray(x, dtype=float).flatten()
    f1, p1, f2, p2, f3, l3d, l3n, f4, l4d, l4n = map(float, x)

    revenue = 0.0
    grad_revenue = np.zeros(10, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        T = E_day + E_night
        w_s, R_s, g_s, B_reg_s = w_data[s], R_data[s], g_levels[s], B_reg_data[s]

        b1, b2 = f1 + p1 * T, f2 + p2 * T
        b3 = f3 + l3d * E_day + l3n * E_night
        b4 = f4 + l4d * E_day + l4n * E_night
        bills = np.array([b1, b2, b3, b4], dtype=float)

        green_adjust = delta * g_s * B_reg_s
        DU_contracts = (bills - green_adjust) - R_s
        DU = np.concatenate(([0.0], DU_contracts))

        u = -beta * DU
        P = softmax_stable(u)

        revenue += w_s * float(np.dot(bills, P[1:]))

        def dr_direction(dbills: np.ndarray) -> float:
            dbills = np.asarray(dbills, dtype=float).reshape(4,)
            dDU = np.concatenate(([0.0], dbills))
            du = -beta * dDU
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return float(np.dot(dbills, P[1:]) + np.dot(bills, dP[1:]))

        grad_revenue[0] += w_s * dr_direction([1.0, 0.0, 0.0, 0.0])   # f1
        grad_revenue[1] += w_s * dr_direction([T,   0.0, 0.0, 0.0])   # p1
        grad_revenue[2] += w_s * dr_direction([0.0, 1.0, 0.0, 0.0])   # f2
        grad_revenue[3] += w_s * dr_direction([0.0, T,   0.0, 0.0])   # p2
        grad_revenue[4] += w_s * dr_direction([0.0, 0.0, 1.0, 0.0])   # f3
        grad_revenue[5] += w_s * dr_direction([0.0, 0.0, E_day, 0.0]) # l3d
        grad_revenue[6] += w_s * dr_direction([0.0, 0.0, E_night, 0.0]) # l3n
        grad_revenue[7] += w_s * dr_direction([0.0, 0.0, 0.0, 1.0])   # f4
        grad_revenue[8] += w_s * dr_direction([0.0, 0.0, 0.0, E_day]) # l4d
        grad_revenue[9] += w_s * dr_direction([0.0, 0.0, 0.0, E_night]) # l4n

    return revenue, grad_revenue

def U_x(x: np.ndarray) -> float:
    rev, _ = profit_and_grad(x)
    return -float(rev)

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(x)
    return -g

# ============================================================
# Scaling (Based on Competitor A)
# ============================================================
SCALE = np.array([
    F_A, flat_A, F_A, flat_A, F_A, tou_A[0], tou_A[1], F_A, tou_A[0], tou_A[1]
], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def to_z(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / SCALE

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))

# ============================================================
# Bounds in z-space
# x = (f1,p1, f2,p2, f3,l3d,l3n, f4,l4d,l4n)
# Flat fees (indices 0,2,4,7) and per-unit rates (rest) bounded separately
# ============================================================
lower = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float)
upper = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float)
bounds = Bounds(lower, upper)

# ============================================================
# Step-taking class that respects bounds
# ============================================================
class RandomDisplacementBounds:
    def __init__(self, xmin, xmax, stepsize=0.15):
        self.xmin = np.asarray(xmin, dtype=float)
        self.xmax = np.asarray(xmax, dtype=float)
        self.stepsize = stepsize

    def __call__(self, x):
        x_new = np.asarray(x, dtype=float) + np.random.uniform(
            -self.stepsize, self.stepsize, size=x.shape
        )
        return np.clip(x_new, self.xmin, self.xmax)

# ============================================================
# Basinhopping runner
# ============================================================
def run_basinhopping(seed: int = 0, niter: int = 200, stepsize: float = 0.20) -> tuple:
    np.random.seed(seed)

    # Start from z=1 (i.e. x = SCALE, matching competitor prices)
    z0 = np.ones(10, dtype=float)

    minimizer_kwargs = {
        "method": "L-BFGS-B",
        "jac": dU_z,
        "bounds": bounds,
    }

    take_step = RandomDisplacementBounds(lower, upper, stepsize=stepsize)

    start = time.perf_counter()

    result = basinhopping(
        func=U_z,
        x0=z0,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        take_step=take_step,
        seed=seed,
        disp=False,
    )

    end = time.perf_counter()
    runtime = end - start

    z_best = result.x
    x_best = to_x(z_best)
    profit_best, _ = profit_and_grad(x_best)

    return result, z_best, x_best, profit_best, runtime

# ============================================================
# Main: repeated runs for mean and std
# ============================================================
if __name__ == "__main__":
    N_RUNS   = 20
    NITER    = 200
    STEPSIZE = 0.20

    profits  = []
    runtimes = []

    best_overall_profit = -np.inf
    best_overall_x      = None
    best_overall_z      = None
    best_overall_result = None

    labels = [
        "f1", "p1",
        "f2", "p2",
        "f3", "l3_day", "l3_night",
        "f4", "l4_day", "l4_night",
    ]

    print("\nRunning basinhopping on extended green model...\n")

    for seed in range(N_RUNS):
        result, z_best, x_best, profit_best, runtime = run_basinhopping(
            seed=seed,
            niter=NITER,
            stepsize=STEPSIZE,
        )

        profits.append(profit_best)
        runtimes.append(runtime)

        if profit_best > best_overall_profit:
            best_overall_profit = profit_best
            best_overall_x      = x_best
            best_overall_z      = z_best
            best_overall_result = result

        print(f"Run {seed+1:2d}: profit = {profit_best:.6f}, runtime = {runtime:.4f} s")

    profits  = np.array(profits)
    runtimes = np.array(runtimes)

    print("\n========== BASINHOPPING SUMMARY ==========\n")
    print("Best overall z-values:")
    for lbl, zv in zip(labels, best_overall_z):
        print(f"  {lbl:10s} z = {zv:.6f}  (x = {to_x(best_overall_z)[list(labels).index(lbl)]:.6f})")

    print(f"\nBest overall profit : {best_overall_profit:.6f}")
    print(f"Best overall U      : {best_overall_result.fun:.6f}")

    print(f"\nMean profit         : {np.mean(profits):.6f}")
    print(f"Std  profit         : {np.std(profits):.6f}")

    print(f"\nMean runtime (s)    : {np.mean(runtimes):.4f}")
    print(f"Std  runtime (s)    : {np.std(runtimes):.4f}")
    print("\n==========================================\n")