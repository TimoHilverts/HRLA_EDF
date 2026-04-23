# postprocessing_BasicModel_HARDCODED.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import BasicModel as model  # <-- your baseline model file (defines U_x, SCALE, etc.)

# ============================================================
# HARDCODED POSTPROCESSING (NO PICKLE) — BASIC MODEL
# - Uses hardcoded solution from Habrok
# - Computes and/or verifies revenue using model.U_x
# - Plots 1D revenue cuts around the hardcoded x*
# ============================================================

# ----------------------------
# Choose how you want to input the solution
# ----------------------------
USE_Z_SPACE = False  # True: provide z* and use model.SCALE, False: provide x* directly

# ----------------------------
# Hardcode Habrok solution
# ----------------------------
# Option A: solution in z-space (scaled variables)
# (Fill this with the "Best value for a=..." vector printed by Habrok if that was z-space.)
z_habrok = None
# np.array([1.0243072,  0.9428475,  0.64082089], dtype=float)  # <-- REPLACE with Habrok z*

# Option B: solution in x-space (original tariffs)
# x = [p, l2_day, l2_night]
x_habrok =  np.array([0.332865, 0.348804, 0.178977], dtype=float) # e.g. np.array([0.345, 0.360, 0.155], dtype=float)

# ----------------------------
# Hardcode Habrok revenue (optional)
# ----------------------------
# Set to None if you want to use the recomputed revenue only.
revenue_habrok_reported = None
#1272.2945  # e.g. 1042.1121

# ----------------------------
# Variable names + plot ranges (in x-space)
# ----------------------------
var_names = ["p", "l2_day", "l2_night"]

# Adjust to the ranges you want for the baseline plots
cut_ranges = {
    # "p": (0.30, 0.35),
    # "l2_day": (0.32, 0.36),
    # "l2_night": (0.15, 0.20),
    "p": (-20, 20),
    "l2_day": (-20, 20),
    "l2_night": (-20, 20),
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
            "BasicModel has no SCALE attribute. In your BasicModel.py you DO define SCALE, "
            "so this likely means you imported the wrong module/file name. "
            "Either fix the import, or set USE_Z_SPACE=False and provide x_habrok directly."
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
print("Hardcoded Habrok solution (BasicModel) — no pickle")
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
               label=f"x* = {x_star[i]:.4f}")

    if np.isfinite(t_max) and np.isfinite(max_rev_1d):
        ax.axvline(x=t_max, color="green", linestyle="--",
                   label=f"1D cut opt = {t_max:.4f}")

    # Mark the (hardcoded) solution point using the recomputed revenue (most consistent)
    ax.scatter(float(x_star[i]), float(revenue_computed), color="red", marker="o", s=50)

    ax.set_title(f"Cut: {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Revenue")
    ax.grid(True)
    ax.legend(fontsize="small", loc="lower left")

fig.suptitle(
    f"1D Revenue Cuts Around Hardcoded Habrok Solution — BasicModel (Rev {revenue_note}={revenue_star:.4f})",
    fontsize=14
)
plt.tight_layout(rect=[0, 0.05, 1, 0.90])

plot_path = output_dir / (
    "1Dcuts_LARGETVALS_BasicModel_M=10_a=20_K=2000_HABROK_HARDCODED_"
    f"beta={getattr(model, 'beta', 'NA')}_Rev{revenue_star:.2f}_allinone.png"
)

plt.savefig(plot_path, dpi=300)
plt.close(fig)

print(f"Saved plot: {plot_path}")
print("Done.")

# ============================================================
# EXTRA FIGURE — 2D contour slice of revenue (to visualise nonconvex structure)
# Fix one variable at x* and grid over the other two.
#
# Drop this at the end of your hardcoded 1D-cuts script (or run as a standalone block),
# after x_star and revenue_computed are defined.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- settings ---
output_dir = Path("output/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Choose which pair to plot:
#   pair = ("p", "l2_day") or ("p", "l2_night") or ("l2_day", "l2_night")
pair = ("l2_day", "l2_night")

# Grid resolution (higher = smoother but slower)
Ngrid = 200

# Ranges (in x-space). Use "wide but economically meaningful" ranges.
# You can tweak these; these are safe defaults around your x_star.
ranges = {
    "p":       (max(0.0, float(x_star[0]) - 0.08), float(x_star[0]) + 0.08),
    "l2_day":  (max(0.0, float(x_star[1]) - 0.08), float(x_star[1]) + 0.08),
    "l2_night":(max(0.0, float(x_star[2]) - 0.08), float(x_star[2]) + 0.08),
}

# Which variable to hold fixed at x*
fixed_name = ({"p", "l2_day", "l2_night"} - set(pair)).pop()
fixed_idx = {"p": 0, "l2_day": 1, "l2_night": 2}[fixed_name]
fixed_val = float(x_star[fixed_idx])

# indices for the two plotted variables
idx = {"p": 0, "l2_day": 1, "l2_night": 2}
i1, i2 = idx[pair[0]], idx[pair[1]]

# grid
a_lo, a_hi = ranges[pair[0]]
b_lo, b_hi = ranges[pair[1]]

A = np.linspace(a_lo, a_hi, Ngrid)
B = np.linspace(b_lo, b_hi, Ngrid)
AA, BB = np.meshgrid(A, B, indexing="xy")

# evaluate revenue on grid
Z = np.full_like(AA, np.nan, dtype=float)

for r in range(Ngrid):
    for c in range(Ngrid):
        x_temp = np.array(x_star, dtype=float)
        x_temp[i1] = AA[r, c]
        x_temp[i2] = BB[r, c]
        # fixed variable stays at x*
        rev = -model.U_x(x_temp)
        Z[r, c] = rev if np.isfinite(rev) else np.nan

# locate max on the grid (ignoring nans)
if np.any(np.isfinite(Z)):
    rr, cc = np.unravel_index(np.nanargmax(Z), Z.shape)
    a_best, b_best, z_best = float(AA[rr, cc]), float(BB[rr, cc]), float(Z[rr, cc])
else:
    a_best, b_best, z_best = np.nan, np.nan, np.nan

# plot
fig = plt.figure(figsize=(7.5, 5.8))
ax = fig.add_subplot(111)

# Filled contours + contour lines
cf = ax.contourf(AA, BB, Z, levels=30)
cs = ax.contour(AA, BB, Z, levels=10, linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8)

cbar = fig.colorbar(cf, ax=ax)
cbar.set_label("Revenue")

# mark x* projection
ax.scatter(float(x_star[i1]), float(x_star[i2]), marker="o", s=60, label="Projection of $x^*$")

# mark best point found on this 2D slice grid
if np.isfinite(a_best) and np.isfinite(b_best):
    ax.scatter(a_best, b_best, marker="x", s=80, label="Best on 2D slice grid")

ax.set_xlabel(pair[0])
ax.set_ylabel(pair[1])
ax.set_title(
    f"2D revenue slice fixing {fixed_name} = {fixed_val:.4f}\n"
    f"(baseline, evaluated around $x^*$)"
)
ax.legend(loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = output_dir / f"Fig_Baseline_2DContour_{pair[0]}_vs_{pair[1]}_fix_{fixed_name}.png"
plt.savefig(out_path, dpi=300)
plt.close(fig)

print(f"Saved 2D contour: {out_path}")
if np.isfinite(z_best):
    print(f"Best on grid slice: {pair[0]}={a_best:.6f}, {pair[1]}={b_best:.6f}, revenue={z_best:.6f}")
