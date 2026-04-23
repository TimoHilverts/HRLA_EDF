# multistart_nonconvexity_demo.py
# ------------------------------------------------------------
# Multi-start experiment to empirically reveal nonconvexity:
# - Sample many random initial points x0 within bounds
# - Run a LOCAL optimizer from each x0 (e.g., L-BFGS-B)
# - Collect final revenues and plot a histogram
#
# Assumes:
#   - BasicModel.py defines U_x(x) where revenue = -U_x(x)
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import BasicModel as model

# You need SciPy for the local solver
from scipy.optimize import minimize

# ----------------------------
# Settings
# ----------------------------
K = 100                   # number of random starts (50-200 typical)
seed = 0                  # reproducibility
maxiter = 500             # local solver iterations
tol = 1e-9                # solver tolerance

# Economically meaningful bounds (EDIT THESE!)
# Format: (lower, upper) for each variable in x = [p, l2_day, l2_night]
bounds = [
    (0.0, 0.6),  # p
    (0.0, 0.6),  # l2_day
    (0.0, 0.6),  # l2_night
]

var_names = ["p", "l2_day", "l2_night"]

output_dir = Path("output/multistart")
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Objective for SciPy
# ----------------------------
# You use revenue = -model.U_x(x). Hence minimizing model.U_x(x)
# corresponds to maximizing revenue.
def objective(x: np.ndarray) -> float:
    val = model.U_x(np.asarray(x, dtype=float))
    # ensure scalar float
    return float(val)

# ----------------------------
# Run multi-start local optimizations
# ----------------------------
rng = np.random.default_rng(seed)
L = np.array([b[0] for b in bounds], dtype=float)
U = np.array([b[1] for b in bounds], dtype=float)

x0s = rng.uniform(L, U, size=(K, 3))

results = []
for k in range(K):
    x0 = x0s[k].copy()

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": tol},
    )

    x_opt = res.x
    u_opt = float(res.fun)                # this is U_x at optimum
    rev_opt = -u_opt                      # revenue

    results.append(
        {
            "k": k,
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(getattr(res, "nfev", -1)),
            "nit": int(getattr(res, "nit", -1)),
            "x_opt": x_opt,
            "U_opt": u_opt,
            "rev_opt": rev_opt,
            "x0": x0,
        }
    )

# Convert to arrays for plotting/analysis
rev = np.array([r["rev_opt"] for r in results], dtype=float)
succ = np.array([r["success"] for r in results], dtype=bool)
Xopt = np.vstack([r["x_opt"] for r in results])

# ----------------------------
# Basic diagnostics
# ----------------------------
best_idx = int(np.nanargmax(rev))
best_rev = float(rev[best_idx])
best_x = Xopt[best_idx]

print("\n====================")
print("Multi-start summary")
print("====================")
print(f"K = {K}, successes = {succ.sum()}/{K}")
print(f"Best revenue  = {best_rev:.6f}")
print("Best x*:")
for name, val in zip(var_names, best_x):
    print(f"  {name:8s} = {val:.6f}")

# ----------------------------
# Plot 1: Histogram of final revenues (core nonconvexity figure)
# ----------------------------
fig = plt.figure(figsize=(7.5, 4.8))
ax = fig.add_subplot(111)

# plot successes only (optional)
rev_plot = rev[np.isfinite(rev)]
ax.hist(rev_plot, bins=25)

ax.set_xlabel("Final revenue after local optimization")
ax.set_ylabel("Count (number of starts)")
ax.set_title("Multi-start convergence: distribution of final revenues")
ax.grid(True, alpha=0.3)

hist_path = output_dir / "Fig_Multistart_Histogram_Revenue.png"
plt.tight_layout()
plt.savefig(hist_path, dpi=300)
plt.close(fig)
print(f"\nSaved histogram: {hist_path}")

# ----------------------------
# Plot 2 (optional): 2D scatter of final points, colored by revenue
# Useful to show distinct basins in parameter space.
# ----------------------------
# Choose which two axes to display:
i1, i2 = 1, 2  # l2_day vs l2_night
fig = plt.figure(figsize=(6.5, 5.2))
ax = fig.add_subplot(111)

sc = ax.scatter(Xopt[:, i1], Xopt[:, i2], c=rev, s=35)
ax.scatter(best_x[i1], best_x[i2], marker="x", s=120, label="Best found")

ax.set_xlabel(var_names[i1])
ax.set_ylabel(var_names[i2])
ax.set_title("Final solutions from multi-start (colored by revenue)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Revenue")

scatter_path = output_dir / f"Fig_Multistart_Scatter_{var_names[i1]}_vs_{var_names[i2]}.png"
plt.tight_layout()
plt.savefig(scatter_path, dpi=300)
plt.close(fig)
print(f"Saved scatter:   {scatter_path}")

# ----------------------------
# Optional: Save a compact text summary for thesis appendix
# ----------------------------
# Cluster-like quick check: count "unique" revenues up to tolerance
# (This is a crude proxy for multiple local optima.)
eps = 1e-3
rev_sorted = np.sort(rev_plot)
clusters = [rev_sorted[0]]
for v in rev_sorted[1:]:
    if abs(v - clusters[-1]) > eps:
        clusters.append(v)

summary_path = output_dir / "multistart_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Multi-start summary\n")
    f.write(f"K={K}, successes={succ.sum()}/{K}\n")
    f.write(f"Best revenue={best_rev:.8f}\n")
    f.write("Best x*:\n")
    for name, val in zip(var_names, best_x):
        f.write(f"  {name} = {val:.8f}\n")
    f.write("\nApprox distinct revenue clusters (eps=1e-3):\n")
    for c in clusters:
        f.write(f"  {c:.8f}\n")

print(f"Saved text:      {summary_path}")
print("\nDone.")
