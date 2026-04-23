import numpy as np
import time
import cma

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
# Profit in x-space
# x = [p, l2_day, l2_night]
# ============================================================
def profit(x: np.ndarray) -> float:
    p, l2_day, l2_night = map(float, x)

    total_profit = 0.0

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
        total_profit += w_s * r

    return float(total_profit)

def U_x(x: np.ndarray) -> float:
    return -profit(np.asarray(x, dtype=float))  # minimize negative profit

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

# ============================================================
# Bounds in z-space
# ============================================================
lower = np.array([0.5, 0.5, 0.5], dtype=float)
upper = np.array([2.0, 2.0, 3.0], dtype=float)

# ============================================================
# Single CMA-ES run
# ============================================================
def run_cmaes_single(
    seed: int = 0,
    sigma0: float = 0.30,
    maxfevals: int = 20000,
    popsize: int | None = None,
):
    z0 = np.ones(3, dtype=float)

    options = {
        "bounds": [lower.tolist(), upper.tolist()],
        "seed": seed,
        "verbose": -9,
        "maxfevals": maxfevals,
    }

    if popsize is not None:
        options["popsize"] = popsize

    start_time = time.perf_counter()

    # xbest is the best solution vector found
    # es contains the optimization state/result
    xbest, es = cma.fmin2(
        U_z,
        z0,
        sigma0,
        options,
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time

    z_best = np.asarray(xbest, dtype=float)
    x_best = to_x(z_best)
    profit_best = profit(x_best)

    summary = {
        "runtime_seconds": runtime,
        "best_profit": float(profit_best),
        "best_z": z_best,
        "best_x": x_best,
        "es": es,
        "evaluations": int(es.result.evaluations),
        "iterations": int(es.result.iterations),
        "stop": es.stop(),
    }

    return summary

# ============================================================
# Repeated CMA-ES runs
# ============================================================
def run_cmaes_repeated(
    n_runs: int = 20,
    sigma0: float = 0.30,
    maxfevals: int = 20000,
    popsize: int | None = None,
):
    profits = []
    runtimes = []
    evaluations = []
    iterations = []
    summaries = []

    best_profit = -np.inf
    best_summary = None

    for seed in range(n_runs):
        summary = run_cmaes_single(
            seed=seed,
            sigma0=sigma0,
            maxfevals=maxfevals,
            popsize=popsize,
        )

        profits.append(summary["best_profit"])
        runtimes.append(summary["runtime_seconds"])
        evaluations.append(summary["evaluations"])
        iterations.append(summary["iterations"])
        summaries.append(summary)

        if summary["best_profit"] > best_profit:
            best_profit = summary["best_profit"]
            best_summary = summary

        print(
            f"Run {seed+1:2d}: "
            f"profit = {summary['best_profit']:.6f}, "
            f"runtime = {summary['runtime_seconds']:.4f} s, "
            f"evals = {summary['evaluations']}"
        )

    profits = np.array(profits, dtype=float)
    runtimes = np.array(runtimes, dtype=float)
    evaluations = np.array(evaluations, dtype=int)
    iterations = np.array(iterations, dtype=int)

    result_summary = {
        "profits": profits,
        "runtimes": runtimes,
        "evaluations": evaluations,
        "iterations": iterations,
        "mean_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits, ddof=1)) if len(profits) > 1 else 0.0,
        "best_profit": float(np.max(profits)),
        "worst_profit": float(np.min(profits)),
        "mean_runtime": float(np.mean(runtimes)),
        "std_runtime": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
        "mean_evaluations": float(np.mean(evaluations)),
        "mean_iterations": float(np.mean(iterations)),
        "best_summary": best_summary,
        "all_summaries": summaries,
    }

    return result_summary

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\nRunning CMA-ES...\n")

    repeated_summary = run_cmaes_repeated(
        n_runs=20,
        sigma0=0.30,
        maxfevals=20000,
        popsize=None,   # leave default first
    )

    best = repeated_summary["best_summary"]

    print("\n=============== CMA-ES RESULT ===============\n")
    print("Number of runs         :", len(repeated_summary["profits"]))
    print("Mean profit (€)        :", repeated_summary["mean_profit"])
    print("Std profit (€)         :", repeated_summary["std_profit"])
    print("Best profit (€)        :", repeated_summary["best_profit"])
    print("Worst profit (€)       :", repeated_summary["worst_profit"])

    print("\nBest z (scaled)        :", best["best_z"])
    print("Best x (tariffs)       :", best["best_x"])
    print("  Flat price p         :", best["best_x"][0])
    print("  ToU day price        :", best["best_x"][1])
    print("  ToU night price      :", best["best_x"][2])

    print("\nMean runtime (seconds) :", repeated_summary["mean_runtime"])
    print("Std runtime (seconds)  :", repeated_summary["std_runtime"])
    print("Mean evaluations       :", repeated_summary["mean_evaluations"])
    print("Mean iterations        :", repeated_summary["mean_iterations"])

    print("\nBest-run evaluations   :", best["evaluations"])
    print("Best-run iterations    :", best["iterations"])
    print("Best-run stop reason   :", best["stop"])

    print("\n=============================================\n")