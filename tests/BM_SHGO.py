import numpy as np
import time
from scipy.optimize import shgo

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
# Scaling wrapper (z-space), same idea as HRLA
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
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


# ============================================================
# Bounds in z-space
# ============================================================
bounds_z = [
    (0.5, 2.0),   # p / comp_flat
    (0.5, 2.0),   # l2_day / comp_tou_day
    (0.5, 3.0),   # l2_night / comp_tou_night
]

# ============================================================
# Single SHGO run
# ============================================================
def run_shgo_single(
    n: int = 200,
    iters: int = 3,
    sampling_method: str = "simplicial",
):
    start_time = time.perf_counter()

    result = shgo(
        func=U_z,
        bounds=bounds_z,
        iters=iters,
        n=n,
        sampling_method=sampling_method,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "jac": dU_z,
            "bounds": bounds_z,
            "options": {
                "maxiter": 5000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        },
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time

    z_best = np.asarray(result.x, dtype=float)
    x_best = to_x(z_best)
    profit_best, _ = profit_and_grad(x_best)

    summary = {
        "runtime_seconds": runtime,
        "best_profit": float(profit_best),
        "best_z": z_best,
        "best_x": x_best,
        "best_result": result,
        "success": bool(result.success),
        "message": result.message,
        "fun": float(result.fun),
        "nfev": getattr(result, "nfev", None),
        "nlfev": getattr(result, "nlfev", None),
        "nljev": getattr(result, "nljev", None),
        "nit": getattr(result, "nit", None),
    }

    return summary


# ============================================================
# Repeated SHGO runs
# Mostly useful if you vary settings; otherwise deterministic
# ============================================================
def run_shgo_repeated(
    n_runs: int = 5,
    n: int = 200,
    iters: int = 3,
    sampling_method: str = "simplicial",
):
    profits = []
    runtimes = []
    summaries = []

    best_profit = -np.inf
    best_summary = None

    for run in range(n_runs):
        summary = run_shgo_single(
            n=n,
            iters=iters,
            sampling_method=sampling_method,
        )

        profits.append(summary["best_profit"])
        runtimes.append(summary["runtime_seconds"])
        summaries.append(summary)

        if summary["best_profit"] > best_profit:
            best_profit = summary["best_profit"]
            best_summary = summary

        print(
            f"Run {run+1:2d}: "
            f"profit = {summary['best_profit']:.6f}, "
            f"runtime = {summary['runtime_seconds']:.4f} s, "
            f"success = {summary['success']}"
        )

    profits = np.array(profits, dtype=float)
    runtimes = np.array(runtimes, dtype=float)

    result_summary = {
        "profits": profits,
        "runtimes": runtimes,
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
        "mean_runtime": float(np.mean(runtimes)),
        "std_runtime": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
        "best_summary": best_summary,
        "all_summaries": summaries,
    }

    return result_summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Gradient check in x-space
    x_test = to_x(np.ones(3, dtype=float))
    g_analytic = dU_x(x_test)
    g_fd = finite_diff_grad(U_x, x_test, eps=1e-6)

    print("Gradient check at x_test =", x_test)
    print("Analytic dU_x:", g_analytic)
    print("FD dU_x      :", g_fd)
    print("Max abs error:", float(np.max(np.abs(g_analytic - g_fd))))

    print("\nRunning SHGO...\n")

    repeated_summary = run_shgo_repeated(
        n_runs=20,
        n=300,
        iters=5,
        sampling_method="simplicial",
    )

    best = repeated_summary["best_summary"]
    result = best["best_result"]

    print("\n=============== SHGO RESULT ===============\n")
    print("Number of runs         :", len(repeated_summary["profits"]))
    print("Mean profit (€)        :", repeated_summary["mean_profit"])
    print("Std profit (€)         :", repeated_summary["std_profit"])
    print("Best profit (€)        :", best["best_profit"])

    print("\nBest z (scaled)        :", best["best_z"])
    print("Best x (tariffs)       :", best["best_x"])
    print("  Flat price p         :", best["best_x"][0])
    print("  ToU day price        :", best["best_x"][1])
    print("  ToU night price      :", best["best_x"][2])

    print("\nMean runtime (seconds) :", repeated_summary["mean_runtime"])
    print("Std runtime (seconds)  :", repeated_summary["std_runtime"])

    print("\nSuccess                :", best["success"])
    print("Message                :", best["message"])
    print("Best U                 :", best["fun"])
    print("Function evals         :", best["nfev"])
    print("Local func evals       :", best["nlfev"])
    print("Local jac evals        :", best["nljev"])
    print("Iterations             :", best["nit"])

    print("\n===========================================\n")