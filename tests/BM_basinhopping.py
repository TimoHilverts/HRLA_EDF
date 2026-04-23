import numpy as np
import time
from scipy.optimize import basinhopping, Bounds

# ============================================================
# MODEL
# ============================================================

S = 10
beta = 0.01
COMP_FIXED_FEE = 196.0

Es_used = np.array([
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

comp_tou = np.array([0.355, 0.250])
comp_flat = 0.342

w_uniform = np.ones(S) / S

# ============================================================
# Stable softmax
# ============================================================

def softmax_stable(u):
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit + gradient
# ============================================================

def profit_and_grad(x):
    p, l2_day, l2_night = map(float, x)

    profit = 0.0
    grad_profit = np.zeros(3, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]
        T = E_day + E_night

        b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night + COMP_FIXED_FEE
        b_c2 = comp_flat * T + COMP_FIXED_FEE
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night
        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

        u = -beta * b
        P = softmax_stable(u)

        r = b_r1 * P[2] + b_r2 * P[3]
        profit += w_s * r

        db_dp = np.array([0.0, 0.0, T, 0.0], dtype=float)
        db_ld = np.array([0.0, 0.0, 0.0, E_day], dtype=float)
        db_ln = np.array([0.0, 0.0, 0.0, E_night], dtype=float)

        def dr_direction(db_dtheta):
            du = -beta * db_dtheta
            du_bar = float(np.dot(P, du))
            dP = P * (du - du_bar)
            return (
                db_dtheta[2] * P[2] + b_r1 * dP[2]
                + db_dtheta[3] * P[3] + b_r2 * dP[3]
            )

        grad_profit[0] += w_s * dr_direction(db_dp)
        grad_profit[1] += w_s * dr_direction(db_ld)
        grad_profit[2] += w_s * dr_direction(db_ln)

    return profit, grad_profit

def U_x(x):
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)

def dU_x(x):
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# ============================================================
# Scaling
# ============================================================

SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]], dtype=float)

def to_x(z):
    return SCALE * np.asarray(z, dtype=float)

def to_z(x):
    return np.asarray(x, dtype=float) / SCALE

def U_z(z):
    return U_x(to_x(z))

def dU_z(z):
    return SCALE * dU_x(to_x(z))

# ============================================================
# Bounds in z-space
# ============================================================

lower = np.array([0.5, 0.5, 0.5], dtype=float)
upper = np.array([2.0, 2.0, 2.5], dtype=float)
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

def run_basinhopping(seed=0, niter=100, stepsize=0.15):
    np.random.seed(seed)

    z0 = np.ones(3, dtype=float)

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
    N_RUNS = 20
    NITER = 200
    STEPSIZE = 0.20

    profits = []
    runtimes = []

    best_overall_profit = -np.inf
    best_overall_x = None
    best_overall_z = None
    best_overall_result = None

    print("\nRunning basinhopping experiments...\n")

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
            best_overall_x = x_best
            best_overall_z = z_best
            best_overall_result = result

        print(f"Run {seed+1:2d}: profit = {profit_best:.6f}, runtime = {runtime:.4f} s")

    profits = np.array(profits)
    runtimes = np.array(runtimes)

    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    mean_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)

    print("\n========== BASINHOPPING SUMMARY ==========\n")
    print("Best overall z      :", best_overall_z)
    print("Best overall x      :", best_overall_x)
    print("Best overall profit :", best_overall_profit)
    print("Best overall U      :", best_overall_result.fun)

    print("\nMean profit         :", mean_profit)
    print("Std profit          :", std_profit)

    print("\nMean runtime (s)    :", mean_runtime)
    print("Std runtime (s)     :", std_runtime)
    print("\n==========================================\n")