import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import FixedPrices as model  # <-- your Habrok extension model file (defines U_x, SCALE, etc.)

# ============================================================
# HARDCODED POSTPROCESSING (NO PICKLE) — FixedPrices.py
# - Hardcode the Habrok solution (z* or x*)
# - Recompute revenue via model.U_x (and optionally compare to reported revenue)
# - Plot consolidated 1D revenue cuts around the hardcoded x*
# ============================================================

# ----------------------------
# Choose how you want to input the solution
# ----------------------------
USE_Z_SPACE = True  # True: provide z* and use model.SCALE, False: provide x* directly

# ----------------------------
# Hardcode Habrok solution
# ----------------------------
# Option A: solution in z-space (scaled variables)
# variables = (f1, p, f2, l2_day, l2_night) in THIS order
z_habrok = np.array([0.49994879, 0.9580467,  0.50326082, 0.91209604, 0.98590237], dtype=float)  # <-- REPLACE with Habrok z*

# Option B: solution in x-space (original variables)
# x = [f1, p, f2, l2_day, l2_night]
x_habrok = None  # e.g. np.array([120.0, 0.34, 120.0, 0.36, 0.22], dtype=float)

# ----------------------------
# Hardcode Habrok revenue (optional)
# ----------------------------
revenue_habrok_reported = None  # e.g. 1420.1234

# ----------------------------
# Variable names + plot ranges (in x-space)
# ----------------------------
var_names = ["f1", "p", "f2", "l2_day", "l2_night"]

# Choose ranges that make sense for YOUR model.
# I set sensible defaults:
# - fixed fees around competitor fixed fee (= 120) with +- 40
# - prices around competitor tariffs
cut_ranges = {
    "f1": (73.0, 80.0),
    "p": (0.28, 0.33),
    "f2": (73.0, 80.0),
    "l2_day": (0.28, 0.33),
    "l2_night": (0.28, 0.33),
}

output_dir = Path("output/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Resolve SCALE, x*, z*
# ============================================================

SCALE = np.asarray(getattr(model, "SCALE", None), dtype=float) if hasattr(model, "SCALE") else None
if SCALE is None:
    raise RuntimeError("FixedPrices.py does not expose SCALE. Please ensure SCALE is defined at module level.")

if USE_Z_SPACE:
    if z_habrok is None:
        raise ValueError("USE_Z_SPACE=True but z_habrok is None.")
    z_star = np.asarray(z_habrok, dtype=float).flatten()
    if z_star.size != 5:
        raise ValueError(f"Expected z_habrok length 5, got {z_star.size}.")
    if SCALE.size != 5:
        raise ValueError(f"Expected SCALE length 5, got {SCALE.size}.")
    x_star = SCALE * z_star
else:
    if x_habrok is None:
        raise ValueError("USE_Z_SPACE=False but x_habrok is None.")
    x_star = np.asarray(x_habrok, dtype=float).flatten()
    if x_star.size != 5:
        raise ValueError(f"Expected x_habrok length 5, got {x_star.size}.")
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
print("Hardcoded Habrok solution (FixedPrices) — no pickle")
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

print("\nSCALE used:")
for name, val in zip(var_names, SCALE):
    print(f"  {name:8s} scale = {val:.6f}")

# ============================================================
# OPTIONAL DIAGNOSTICS (recommended for high-dimensional models)
# Add this block to the hardcoded postprocessing scripts I gave you,
# right AFTER you have x_star, revenue_computed, var_names, and (optionally) z_star.
# Requires: model.dU_x to exist (it does in your GreenComp / extensions).
# ============================================================

RUN_DIAGNOSTICS = True

if RUN_DIAGNOSTICS:
    print("\n====================")
    print("Diagnostics at hardcoded solution")
    print("====================")

    # 1) Stationarity check: gradient norm at x*
    try:
        g = np.asarray(model.dU_x(x_star), dtype=float).flatten()
        grad_inf = float(np.max(np.abs(g)))
        grad_2 = float(np.linalg.norm(g))
        print(f"||grad U(x*)||_inf = {grad_inf:.6e}")
        print(f"||grad U(x*)||_2   = {grad_2:.6e}")
    except Exception as e:
        print("WARNING: Could not compute gradient diagnostics (model.dU_x missing or failed).")
        print("Reason:", repr(e))

    # 2) Local robustness: random perturbation test around x*
    #    (How often do small random moves improve revenue?)
    rng = np.random.default_rng(0)

    # Relative perturbation sizes per variable type:
    # - fixed fees: 2% relative
    # - prices: 2% relative
    # You can tweak these; for the full model, 0.5%-2% is usually informative.
    rel_fixed = 0.02
    rel_price = 0.02

    # Identify "fixed fee" vs "price-like" vars by name (works with your naming)
    rel = np.array([
        rel_fixed if name.startswith("f") else rel_price
        for name in var_names
    ], dtype=float)

    n_trials = 200
    revenues = []
    xs = []

    base_rev = float(revenue_computed)

    for _ in range(n_trials):
        # multiplicative perturbation: x' = x * (1 + rel * N(0,1))
        noise = rng.normal(0.0, 1.0, size=len(x_star))
        x_try = np.array(x_star, dtype=float) * (1.0 + rel * noise)

        # OPTIONAL: enforce simple sanity bounds to avoid nonsense values
        # (uncomment if you want)
        # for i, name in enumerate(var_names):
        #     if name.startswith("f"):
        #         x_try[i] = np.clip(x_try[i], 0.0, 500.0)
        #     else:
        #         x_try[i] = np.clip(x_try[i], 0.0, 1.0)

        rev_try = float(-model.U_x(x_try))
        revenues.append(rev_try)
        xs.append(x_try)

    revenues = np.asarray(revenues, dtype=float)
    best_local = float(np.nanmax(revenues))
    frac_better = float(np.mean(revenues > base_rev))

    print(f"\nRandom perturbation test (n={n_trials}, rel_fixed={rel_fixed:.3f}, rel_price={rel_price:.3f})")
    print(f"Revenue at x*                 = {base_rev:.8f}")
    print(f"Best revenue among perturbations= {best_local:.8f}")
    print(f"Fraction perturbations better  = {frac_better:.3%}")
    print(f"Best improvement               = {best_local - base_rev:.6f}")

    # If many perturbations improve revenue, show the best x found:
    if best_local > base_rev:
        idx = int(np.nanargmax(revenues))
        x_best_local = xs[idx]
        print("\nBest perturbation x_best_local:")
        for name, val in zip(var_names, x_best_local):
            print(f"  {name:10s} = {val:.6f}")

    # 3) (Optional) 1D-cut gap summary (helps interpret your “slice maxima far away” issue)
    #    This uses the already defined cut_ranges + your 1D sweep settings.
    try:
        print("\n1D-cut gap summary (holding other variables fixed at x*)")
        for i, name in enumerate(var_names):
            lo, hi = cut_ranges[name]
            tvals = np.linspace(lo, hi, 600)  # slightly denser for diagnostics

            revs = []
            for t in tvals:
                x_tmp = np.array(x_star, dtype=float)
                x_tmp[i] = t
                revs.append(float(-model.U_x(x_tmp)))
            revs = np.asarray(revs, dtype=float)

            j = int(np.nanargmax(revs))
            t_max = float(tvals[j])
            rev_max = float(revs[j])

            print(f"  {name:10s}: x*={x_star[i]:.6f}, 1D-opt={t_max:.6f}, "
                  f"rev(x*)={base_rev:.6f}, rev(1D-opt)={rev_max:.6f}, gap={rev_max-base_rev:.6f}")
    except Exception as e:
        print("WARNING: 1D-cut gap diagnostics failed.")
        print("Reason:", repr(e))


# ============================================================
# Plot: consolidated 1D cuts around x*
# ============================================================

print("\nGenerating consolidated 1D cut plot...")

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22, 4))
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
               label=f"Habrok x* = {x_star[i]:.4g}")

    if np.isfinite(t_max) and np.isfinite(max_rev_1d):
        ax.axvline(x=t_max, color="green", linestyle="--",
                   label=f"1D cut opt = {t_max:.4g}")

    ax.scatter(float(x_star[i]), float(revenue_computed), color="red", marker="o", s=40)

    ax.set_title(f"Cut: {name}")
    ax.set_xlabel(name)
    ax.set_ylabel("Revenue")
    ax.grid(True)
    ax.legend(fontsize="x-small", loc="lower left")

fig.suptitle(
    f"1D Revenue Cuts Around Hardcoded Habrok Solution — FixedPrices (Rev {revenue_note}={revenue_star:.4f})",
    fontsize=14
)
plt.tight_layout(rect=[0, 0.05, 1, 0.90])

plot_path = output_dir / (
    "1Dcuts_FixedPrices_BETA=0.02_A=20_K=15000_HABROK_HARDCODED_"
    f"beta={getattr(model, 'beta', 'NA')}_Rev{revenue_star:.2f}_allinone.png"
)

plt.savefig(plot_path, dpi=300)
plt.close(fig)

print(f"Saved plot: {plot_path}")
print("Done.")
