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
x_star =  np.array([0.332865, 0.348804, 0.178977], dtype=float) # e.g. np.array([0.345, 0.360, 0.155], dtype=float)

# ----------------------------
# Hardcode Habrok revenue (optional)
# ----------------------------
# Set to None if you want to use the recomputed revenue only.
revenue_habrok_reported = None
#1272.2945  # e.g. 1042.1121

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
