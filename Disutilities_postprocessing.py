import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import Disutilities as model  # <-- your Habrok extension model file

# ============================================================
# HARDCODED POSTPROCESSING (NO PICKLE) — Disutilities.py
# - Uses hardcoded solution from Habrok
# - Computes and/or verifies revenue using model.U_x
# - Plots 1D revenue cuts around the hardcoded x*
# ============================================================

# ----------------------------
# Choose how you want to input the solution
# ----------------------------
USE_Z_SPACE = True  # True: provide z* (and use model.SCALE), False: provide x* directly

# ----------------------------
# Hardcode Habrok solution
# ----------------------------
# Option A: solution in z-space (scaled variables)
# Replace with the best z-vector printed on Habrok
z_habrok = np.array([0.90767827, 0.88156328, 0.76425183], dtype=float)  # <-- REPLACE with Habrok z*

# Option B: solution in x-space (original tariffs): x = [p, l2_day, l2_night]
x_habrok = None  # e.g. np.array([0.345, 0.360, 0.155], dtype=float)

# ----------------------------
# Hardcode Habrok revenue (optional)
# ----------------------------
# Set to None if you want to use the recomputed revenue only.
revenue_habrok_reported = 1042.1121  # e.g. 1358.1234

# ----------------------------
# Variable names + plot ranges (in x-space)
# ----------------------------
var_names = ["p", "l2_day", "l2_night"]

cut_ranges = {
    "p": (0.28, 0.32),
    "l2_day": (0.28, 0.32),
    "l2_night": (0.21, 0.25),
}

output_dir = Path("output/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Resolve SCALE, x*, z*
# ============================================================

SCALE = np.asarray(getattr(model, "SCALE", None), dtype=float) if hasattr(model, "SCALE") else None

if USE_Z_SPACE:
    if z_habrok is None:
        raise ValueError("USE_Z_SPACE=True but z_habrok is None.")
    z_star = np.asarray(z_habrok, dtype=float).flatten()
    if z_star.size != 3:
        raise ValueError(f"Expected z_habrok length 3, got {z_star.size}.")

    if SCALE is None:
        raise ValueError(
            "Model has no SCALE attribute. Either add SCALE to Disutilities.py, "
            "or set USE_Z_SPACE=False and provide x_habrok directly."
        )
    if SCALE.size != 3:
        raise ValueError(f"Expected SCALE length 3, got {SCALE.size}.")

    x_star = SCALE * z_star
else:
    if x_habrok is None:
        raise ValueError("USE_Z_SPACE=False but x_habrok is None.")
    x_star = np.asarray(x_habrok, dtype=float).flatten()
    if x_star.size != 3:
        raise ValueError(f"Expected x_habrok length 3, got {x_star.size}.")
    z_star = None

# ============================================================
# Revenue: reported vs recomputed
# ============================================================

revenue_computed = -model.U_x(x_star)

if revenue_habrok_reported is None:
    revenue_star = float(revenue_computed)
    revenue_note = "computed"
else:
    revenue_star = float(revenue_habrok_reported)
    revenue_note = "reported"
    print(f"NOTE: Habrok revenue (reported) = {revenue_star:.6f}")
    print(f"NOTE: Revenue recomputed from current model = {revenue_computed:.6f}")
    print(f"NOTE: Absolute difference = {abs(revenue_computed - revenue_star):.6e}")

# ============================================================
# Print results
# ============================================================

print("\n====================")
print("Hardcoded Habrok solution (Disutilities) — no pickle")
print(f"Revenue ({revenue_note}) = {revenue_star:.12f}")
print("====================\n")

if z_star is not None:
    print("Optimal vector BEFORE scaling (z*):")
    for name, val in zip(var_names, z_star):
        print(f"  {name:8s} = {val:.6f}")
    print()

print("Optimal vector AFTER scaling / in original variables (x*):")
for name, val in zip(var_names, x_star):
    print(f"  {name:8s} = {val:.6f}")

if SCALE is not None:
    print("\nSCALE used:")
    for name, val in zip(var_names, SCALE):
        print(f"  {name:8s} scale = {val:.6f}")

# ============================================================
# Plot: consolidated 1D cuts around x*
# ============================================================

print("\nGenerating consolidated 1D cut plot...")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
axes = np.atleast_1d(axes).flatten()

for i, name in enumerate(var_names):
    ax = axes[i]
    revenues = []

    lo, hi = cut_ranges[name]
    tvals = np.linspace(lo, hi, 400)

    for t in tvals:
        x_temp = np.array(x_star, dtype=float)
        x_temp[i] = t
        rev = -model.U_x(x_temp)
        revenues.append(rev if np.isfinite(rev) else np.nan)

    revenues = np.asarray(revenues, dtype=float)

    if np.any(np.isfinite(revenues)):
        idx_max = int(np.nanargmax(revenues))
        t_max = float(tvals[idx_max])
        max_rev_1d = float(revenues[idx_max])
    else:
        t_max = np.nan
        max_rev_1d = np.nan

    ax.plot(tvals, revenues, color="black", label="Revenue along 1D cut")

    ax.axvline(x=float(x_star[i]), color="red", linestyle="--",
               label=f"Habrok x* = {x_star[i]:.4f}")

    if np.isfinite(t_max) and np.isfinite(max_rev_1d):
        ax.axvline(x=t_max, color="green", linestyle="--",
                   label=f"1D cut opt = {t_max:.4f}")

    ax.scatter(float(x_star[i]), float(revenue_computed), color="red", marker="o", s=50)

    ax.set_title(f"Cut: {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Revenue")
    ax.grid(True)
    ax.legend(fontsize="small", loc="lower left")

fig.suptitle(
    f"1D Revenue Cuts Around Hardcoded Habrok Solution — Disutilities (Rev {revenue_note}={revenue_star:.4f})",
    fontsize=14
)
plt.tight_layout(rect=[0, 0.05, 1, 0.90])

plot_path = output_dir / (
    "1Dcuts_Disutilities_HABROK_HARDCODED_K=2.000"
    f"beta={getattr(model, 'beta', 'NA')}_Rev{revenue_star:.2f}_allinone.png"
)

plt.savefig(plot_path, dpi=300)
plt.close(fig)

print(f"Saved plot: {plot_path}")
print("Done.")
