# postprocessing_FixedPrices_HARDCODED_minimal_outputs.py
#
# Minimal postprocessing for the FIXED-FEE retailer model ("FixedPrices.py").
# Produces (in ONE script):
#   1) LaTeX table: optimal contract menu + total expected revenue
#   2) Figure: heatmap of segment-level choice probabilities
#   3) Figure: bar plot of aggregate market shares (Outside vs Contract 1 vs Contract 2)
#
# MODEL-SPECIFIC DETAILS (FixedPrices.py):
#   - Decision variables: x = (f1, p, f2, l2_day, l2_night)
#   - Competitors (embedded in R_data) include annual fixed fees
#   - Reservation bill per segment is R_s = min competitor TOTAL bill (var + fixed)
#   - Disutilities: DU_outside = 0, DU_n = bill_n - R_s
#   - Logit is applied over disutilities (NOT directly over bills)
#
# Assumes your model file is importable, e.g. saved as:
#   FixedPrices.py
# and defines at module scope:
#   Es_used (S,2), beta, w_data, R_data
# Optionally also:
#   SCALE, to_x(z)
#
# No pickles used. Uses hardcoded optimal solution (either x* or z*).
#
# Outputs:
#   output/tables/Table_FixedPrices_OptimalContractMenu_WithRevenue.tex
#   output/plots/Fig_FixedPrices_ChoiceProbabilities_Heatmap.png
#   output/plots/Fig_FixedPrices_MarketShares_Bar.png

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# CHANGE THIS IMPORT NAME to match your saved model filename
# Example: if your model is saved as "FixedPrices.py", keep as-is.
# ------------------------------------------------------------
import FixedPrices as model

# ----------------------------
# Choose how you want to input the solution
# ----------------------------
USE_Z_SPACE = True  # True: provide z* and convert via model.to_x(z); False: provide x* directly

# Option A: hardcode best z* (scaled variables) from HRLA output
# z = (f1, p, f2, l2_day, l2_night) in scaled coordinates
z_star = np.array([0.49994879, 0.9580467,  0.50326082, 0.91209604, 0.98590237]
, dtype=float)  # <-- REPLACE with your best z*

# Option B: hardcode best x* (original variables): x = (f1, p, f2, l2_day, l2_night)
# x_star = np.array([150.0, 0.34, 150.0, 0.37, 0.28], dtype=float)  # <-- REPLACE if USE_Z_SPACE=False

# ----------------------------
# Output dirs
# ----------------------------
plots_dir = Path("output/plots")
tables_dir = Path("output/tables")
plots_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Resolve x*
# ============================================================
if USE_Z_SPACE:
    if z_star is None:
        raise ValueError("USE_Z_SPACE=True but z_star is None.")
    if not hasattr(model, "to_x"):
        raise ValueError("USE_Z_SPACE=True but model.to_x(z) is not defined in your FixedPrices model module.")
    x_star = np.asarray(model.to_x(np.asarray(z_star, dtype=float)), dtype=float).flatten()
else:
    if "x_star" not in globals() or x_star is None:
        raise ValueError("USE_Z_SPACE=False but x_star is None.")
    x_star = np.asarray(x_star, dtype=float).flatten()

if x_star.size != 5:
    raise ValueError(f"Expected x_star length 5 (f1, p, f2, l2_day, l2_night), got {x_star.size}.")

f1_val, p_val, f2_val, l2d_val, l2n_val = map(float, x_star)

# ============================================================
# Load FixedPrices primitives
# ============================================================
Es = np.asarray(model.Es_used, dtype=float)                 # (S,2)
beta = float(model.beta)
w = np.asarray(model.w_data, dtype=float).flatten()         # (S,)
R = np.asarray(model.R_data, dtype=float).flatten()         # (S,) reservation bills (incl comp fixed fees)

S = Es.shape[0]
if Es.ndim != 2 or Es.shape[1] != 2:
    raise ValueError(f"Expected Es_used with shape (S,2), got {Es.shape}.")
if w.size != S:
    raise ValueError(f"w_data length {w.size} does not match number of segments S={S}.")
if R.size != S:
    raise ValueError(f"R_data length {R.size} does not match number of segments S={S}.")

# Ensure weights sum to 1 (model already normalizes, but safe)
wsum = float(np.sum(w))
if not np.isfinite(wsum) or wsum <= 0:
    raise ValueError("w_data must be positive and sum to a finite positive value.")
w = w / wsum

# ============================================================
# Bills and probabilities at x*
# Alternatives: [Outside, Contract 1 (Flat), Contract 2 (ToU)]
# Outside disutility is 0 by model definition.
# Contract disutility: DU_n = bill_n - R_s.
# ============================================================
E_day = Es[:, 0]
E_night = Es[:, 1]

bill_c1 = f1_val + p_val * (E_day + E_night)
bill_c2 = f2_val + l2d_val * E_day + l2n_val * E_night

DU_out = np.zeros(S, dtype=float)
DU_c1 = bill_c1 - R
DU_c2 = bill_c2 - R

DU = np.column_stack([DU_out, DU_c1, DU_c2])  # (S,3)

# logit over disutilities
Eexp = np.exp(-beta * DU)
P = Eexp / Eexp.sum(axis=1, keepdims=True)  # (S,3)

# Expected revenue (outside yields 0)
rev_seg = w * (bill_c1 * P[:, 1] + bill_c2 * P[:, 2])
rev_total = float(np.sum(rev_seg))

# Aggregate market shares
share_out = float(np.sum(w * P[:, 0]))
share_c1 = float(np.sum(w * P[:, 1]))
share_c2 = float(np.sum(w * P[:, 2]))

# ============================================================
# 1) LaTeX table: optimal menu + total revenue
# ============================================================
f14, p4, f24, l2d4, l2n4 = (round(f1_val, 4), round(p_val, 4), round(f2_val, 4), round(l2d_val, 4), round(l2n_val, 4))
rev2 = round(rev_total, 2)

table_tex = rf"""
\begin{{table}}[ht]
\centering
\caption{{Optimal tariff parameters under the fixed-fee model with reservation bills and disutility-based choice.}}
\label{{tab:fixedprices_contract_menu_with_revenue}}
\begin{{tabular}}{{lcc}}
\toprule
 & Contract 1 (Flat) & Contract 2 (ToU) \\
\midrule
Fixed fee (€/year)  & {f14:.4f} & {f24:.4f} \\
Peak (€/kWh)        & {p4:.4f}  & {l2d4:.4f} \\
Off-peak (€/kWh)    & {p4:.4f}  & {l2n4:.4f} \\
\midrule
\multicolumn{{3}}{{l}}{{\textbf{{Total expected revenue (€):}} {rev2:.2f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()

table_path = tables_dir / "Table_FixedPrices_OptimalContractMenu_WithRevenue.tex"
table_path.write_text(table_tex, encoding="utf-8")
print(f"Saved: {table_path}")

# ============================================================
# 2) Figure: heatmap of choice probabilities
# ============================================================
fig = plt.figure(figsize=(7.2, 4.2))
ax = fig.add_subplot(111)
im = ax.imshow(P, aspect="auto")
ax.set_yticks(np.arange(S))
ax.set_yticklabels([f"{i}" for i in range(1, S + 1)])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Outside (DU=0)", "Contract 1", "Contract 2"])
ax.set_xlabel("Alternative")
ax.set_ylabel("Segment")
ax.set_title("Segment-level choice probabilities at $x^*$ (fixed-fee model)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Probability")
plt.tight_layout()

heatmap_path = plots_dir / "Fig_FixedPrices_ChoiceProbabilities_Heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close(fig)
print(f"Saved: {heatmap_path}")

# ============================================================
# 3) Figure: aggregate market shares (bar plot)
# ============================================================
labels = ["Outside", "Contract 1", "Contract 2"]
shares = np.array([share_out, share_c1, share_c2], dtype=float)

fig = plt.figure(figsize=(6.8, 3.8))
ax = fig.add_subplot(111)
ax.bar(labels, shares)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("Expected market share")
ax.set_title("Aggregate market shares at $x^*$ (fixed-fee model)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

ms_path = plots_dir / "Fig_FixedPrices_MarketShares_Bar.png"
plt.savefig(ms_path, dpi=300)
plt.close(fig)
print(f"Saved: {ms_path}")

# Optional: quick console summary
print("\n====================")
print("FixedPrices postprocessing summary")
print("====================")
print(f"x* = [f1, p, f2, l2_day, l2_night] = [{f1_val:.6f}, {p_val:.6f}, {f2_val:.6f}, {l2d_val:.6f}, {l2n_val:.6f}]")
print(f"Total expected revenue = {rev_total:.6f}")
print(f"Market shares: Outside={share_out:.4f}, C1={share_c1:.4f}, C2={share_c2:.4f}")
print("Done.")
