import numpy as np
import time
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

# ============================================================
# MODEL:
# Retailer contracts (2):
#   (R1) flat: p
#   (R2) ToU : (l2_day, l2_night)
#
# Competitor contracts (2) -> outside options:
#   (C1) ToU : (0.355, 0.250) + Fixed Fee 196
#   (C2) flat: 0.342 + Fixed Fee 196
# ============================================================

S = 10
beta = 5
COMP_FIXED_FEE = 196.0

# --------------------------
# Consumption (kWh)
# --------------------------
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

# --------------------------
# Competitors
# --------------------------
comp_tou = np.array([0.355, 0.250], dtype=float)
comp_flat = 0.342

# Uniform weights
w_uniform = np.ones(S, dtype=float) / S

# ============================================================
# Stable softmax
# ============================================================
def softmax_stable(u: np.ndarray) -> np.ndarray:
    m = np.max(u)
    ex = np.exp(u - m)
    return ex / np.sum(ex)

# ============================================================
# Profit + analytic gradient (x-space)
# x = [p, l2_day, l2_night]
# ============================================================
def profit_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
    p, l2_day, l2_night = map(float, x)

    profit = 0.0
    grad_profit = np.zeros(3, dtype=float)

    for s in range(S):
        E_day, E_night = Es_used[s]
        w_s = w_uniform[s]
        T = E_day + E_night

        # Bills
        b_c1 = comp_tou[0] * E_day + comp_tou[1] * E_night + COMP_FIXED_FEE
        b_c2 = comp_flat * T + COMP_FIXED_FEE
        b_r1 = p * T
        b_r2 = l2_day * E_day + l2_night * E_night
        b = np.array([b_c1, b_c2, b_r1, b_r2], dtype=float)

        # Logit probabilities
        u = -beta * b
        P = softmax_stable(u)

        # Expected retailer revenue
        r = b_r1 * P[2] + b_r2 * P[3]
        profit += w_s * r

        # Derivatives
        db_dp = np.array([0.0, 0.0, T, 0.0], dtype=float)
        db_ld = np.array([0.0, 0.0, 0.0, E_day], dtype=float)
        db_ln = np.array([0.0, 0.0, 0.0, E_night], dtype=float)

        def dr_direction(db_dtheta: np.ndarray) -> float:
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

def U_x(x: np.ndarray) -> float:
    prof, _ = profit_and_grad(np.asarray(x, dtype=float))
    return -float(prof)  # minimize negative profit

def dU_x(x: np.ndarray) -> np.ndarray:
    _, g = profit_and_grad(np.asarray(x, dtype=float))
    return -g

# ============================================================
# Scaling wrapper (z-space), same as your HRLA setup
# x = SCALE * z
# ============================================================
SCALE = np.array([comp_flat, comp_tou[0], comp_tou[1]], dtype=float)

def to_x(z: np.ndarray) -> np.ndarray:
    return SCALE * np.asarray(z, dtype=float)

def to_z(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) / SCALE

def U_z(z: np.ndarray) -> float:
    return U_x(to_x(z))

def dU_z(z: np.ndarray) -> np.ndarray:
    return SCALE * dU_x(to_x(z))

# ============================================================
# Gradient checker
# ============================================================
def finite_diff_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g

# ============================================================
# HRLA initializer
# ============================================================
d = 3
title = "BasicModel_N2_2competitors_FixedFee_billlogit_uniformweights_scaled_numpy_HRLA"

def initial() -> np.ndarray:
    z0 = np.ones(d, dtype=float)
    z0 += np.random.normal(0.0, 0.2, size=d)
    return z0

# ============================================================
# Helper to extract best z from PostProcessor.get_best(...)
# You may need to adapt this slightly depending on the object returned.
# ============================================================
def extract_best_z(get_best_output) -> np.ndarray:
    """
    Try to extract the best z-vector from PostProcessor.get_best(...).
    Adjust this function if your get_best output has a different structure.
    """
    # Case 1: output is already a vector-like object
    arr = np.asarray(get_best_output, dtype=float)
    if arr.ndim == 1 and arr.size == d:
        return arr

    # Case 2: output is a dict containing z
    if isinstance(get_best_output, dict):
        if "z" in get_best_output:
            return np.asarray(get_best_output["z"], dtype=float)
        if "best_z" in get_best_output:
            return np.asarray(get_best_output["best_z"], dtype=float)
        if "argmin" in get_best_output:
            return np.asarray(get_best_output["argmin"], dtype=float)
        if "x" in get_best_output:
            candidate = np.asarray(get_best_output["x"], dtype=float)
            if candidate.size == d:
                return candidate

    # Case 3: output is list/tuple and one element is the z-vector
    if isinstance(get_best_output, (list, tuple)):
        for item in get_best_output:
            try:
                candidate = np.asarray(item, dtype=float)
                if candidate.ndim == 1 and candidate.size == d:
                    return candidate
            except Exception:
                pass

    raise ValueError(
        "Could not extract best z from postprocessor.get_best(...). "
        "Please inspect the returned object and adapt extract_best_z(...)."
    )

# ============================================================
# Single HRLA run with timing
# ============================================================
def run_hrla(
    M: int = 10,
    N: int = 1,
    K: int = 5000,
    h: float = 1e-9,
    As: list[int] = [1, 3, 5, 10, 15, 20],
    measured: list[int] = [5000],
    sim_annealing: bool = True,
):
    start_time = time.perf_counter()

    algorithm = GO.HRLA(
        d=d,
        M=M,
        N=N,
        K=K,
        h=h,
        title=title,
        U=U_z,
        dU=dU_z,
        initial=initial,
    )

    samples_filename = algorithm.generate_samples(
        As=As,
        sim_annealing=sim_annealing
    )

    postprocessor = PostProcessor(samples_filename)
    postprocessor.compute_tables(measured, 1, "best")
    best_output = postprocessor.get_best(measured=measured, dpi=1)

    end_time = time.perf_counter()
    runtime = end_time - start_time

    # Try to extract best z from get_best output
    z_best = extract_best_z(best_output)
    x_best = to_x(z_best)
    profit_best, _ = profit_and_grad(x_best)

    # Very rough evaluation count proxy, if you want one:
    # HRLA often scales roughly with M*K objective/gradient calls,
    # but this depends on your exact implementation.
    approx_eval_count = M * K

    summary = {
        "runtime_seconds": runtime,
        "samples_filename": samples_filename,
        "best_raw_output": best_output,
        "best_z": z_best,
        "best_x": x_best,
        "best_profit": float(profit_best),
        "approx_eval_count": int(approx_eval_count),
        "M": M,
        "N": N,
        "K": K,
        "h": h,
        "As": As,
        "measured": measured,
    }

    return summary

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Gradient check in x-space
    x_test = to_x(np.ones(d, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    print("\nRunning HRLA...\n")

    summary = run_hrla(
        M=10,
        N=1,
        K=20000,
        h=1e-6,
        As=[10],
        measured=[20000],
        sim_annealing=False,
    )

    print("\n=============== HRLA RESULT ===============\n")
    print("Best z (scaled)         :", summary["best_z"])
    print("Best x (tariffs)        :", summary["best_x"])
    print("  Flat price p          :", summary["best_x"][0])
    print("  ToU day price         :", summary["best_x"][1])
    print("  ToU night price       :", summary["best_x"][2])

    print("\nBest revenue (€)        :", summary["best_profit"])
    print("Runtime (seconds)       :", summary["runtime_seconds"])
    print("Approx. eval count      :", summary["approx_eval_count"])
    print("Samples file            :", summary["samples_filename"])

    print("\n===========================================\n")