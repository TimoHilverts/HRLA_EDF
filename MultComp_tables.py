# postprocessing_MultComp_HARDCODED_minimal_outputs.py
#
# Minimal postprocessing for the MULTI-COMPETITOR + RESERVATION BILL model.
# Produces (in ONE script):
#   1) LaTeX table: optimal contract menu + total expected revenue
#   2) Figure: heatmap of segment-level choice probabilities
#   3) Figure: bar plot of aggregate market shares (Outside vs C1 vs C2)
#
# IMPORTANT differences vs baseline:
#   - Outside option bill is the reservation bill R_s (NOT Es @ competitor tariff)
#   - Uses REAL segment weights w_data (not uniform weights)
#   - Competitors are already “embedded” in R_data (minimum competitor bill)
#
# Assumes your MultComp model file (the one you pasted) is saved as an importable module, e.g.:
#   MultCompModel.py
# and that it defines at module scope:
#   Es_used (S,2), beta, w_data, R_data
# Optionally also:
#   SCALE, to_x(z)
#
# No pickles used. Uses hardcoded optimal solution (either x* or z*).
#
# Outputs:
#   output/tables/Table_MultComp_OptimalContractMenu_WithRevenue.tex
#   output/plots/Fig_MultComp_ChoiceProbabilities_Heatmap.png
#   output/plots/Fig_MultComp_MarketShares_Bar.png

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# CHANGE THIS IMPORT NAME to match your saved model filename
# Example: if you saved the model as "MultCompModel.py", keep this as-is.
# ------------------------------------------------------------
import MultComp_ResBill as model

# ----------------------------
# Choose how you want to input the solution
# ----------------------------
USE_Z_SPACE = True  # True: provide z* and convert via model.to_x(z); False: provide x* directly

# Option A: hardcode best z* (scaled variables) from HRLA output
#z_star = None
z_star = np.array([0.90759828, 0.88196866, 0.76127022], dtype=float)

# Option B: hardcode best x* (original tariffs): x = [p, l2_day, l2_night]
# Fill this with YOUR best x* for the MultComp model.
#x_star = np.array([0.90759828, 0.88196866, 0.76127022], dtype=float)  # <-- REPLACE

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
        raise ValueError("USE_Z_SPACE=True but model.to_x(z) is not defined in your MultComp model module.")
    x_star = np.asarray(model.to_x(np.asarray(z_star, dtype=float)), dtype=float).flatten()
else:
    if x_star is None:
        raise ValueError("USE_Z_SPACE=False but x_star is None.")
    x_star = np.asarray(x_star, dtype=float).flatten()

if x_star.size != 3:
    raise ValueError(f"Expected x_star length 3 (p, l2_day, l2_night), got {x_star.size}.")

p_val, l2d_val, l2n_val = map(float, x_star)

# ============================================================
# Load MultComp primitives
# ============================================================
Es = np.asarray(model.Es_used, dtype=float)                 # (S,2)
beta = float(model.beta)
w = np.asarray(model.w_data, dtype=float).flatten()         # (S,)
R = np.asarray(model.R_data, dtype=float).flatten()         # (S,) reservation bills

S = Es.shape[0]
if Es.ndim != 2 or Es.shape[1] != 2:
    raise ValueError(f"Expected Es_used with shape (S,2), got {Es.shape}.")
if w.size != S:
    raise ValueError(f"w_data length {w.size} does not match number of segments S={S}.")
if R.size != S:
    raise ValueError(f"R_data length {R.size} does not match number of segments S={S}.")

# Ensure weights sum to 1 (your model normalizes already, but safe)
wsum = float(np.sum(w))
if not np.isfinite(wsum) or wsum <= 0:
    raise ValueError("w_data must be positive and sum to a finite positive value.")
w = w / wsum

# ============================================================
# Bills and probabilities at x*
# Alternatives: [Outside, Contract 1 (Flat), Contract 2 (ToU)]
# Outside bill is reservation bill R_s
# ============================================================
bill_out = R
bill_c1 = Es[:, 0] * p_val + Es[:, 1] * p_val
bill_c2 = Es[:, 0] * l2d_val + Es[:, 1] * l2n_val

B = np.column_stack([bill_out, bill_c1, bill_c2])  # (S,3)

# logit over bills
Eexp = np.exp(-beta * B)
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
p4, l2d4, l2n4 = (round(p_val, 4), round(l2d_val, 4), round(l2n_val, 4))
rev2 = round(rev_total, 2)

table_tex = rf"""
\begin{{table}}[ht]
\centering
\caption{{Optimal tariff parameters under the multiple-competitor model with reservation bills.}}
\label{{tab:multcomp_contract_menu_with_revenue}}
\begin{{tabular}}{{lcc}}
\toprule
 & Contract 1 (Flat) & Contract 2 (ToU) \\
\midrule
Peak (€/kWh)      & {p4:.4f} & {l2d4:.4f} \\
Off-peak (€/kWh)  & {p4:.4f} & {l2n4:.4f} \\
\midrule
\multicolumn{{3}}{{l}}{{\textbf{{Total expected revenue (€):}} {rev2:.2f}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()

table_path = tables_dir / "Table_MultComp_OptimalContractMenu_WithRevenue.tex"
table_path.write_text(table_tex, encoding="utf-8")
print(f"Saved: {table_path}")

# ============================================================
# 2) Figure: heatmap of choice probabilities (match baseline sizing)
# ============================================================
plots_dir = Path("output/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# Match baseline: figsize and colorbar sizing
fig = plt.figure(figsize=(10, 4.5))

ax = fig.add_subplot(1, 1, 1)
im = ax.imshow(P, aspect="auto")

ax.set_yticks(np.arange(S))
ax.set_yticklabels([f"{i}" for i in range(1, S + 1)])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Outside (R)", "Contract 1", "Contract 2"])
ax.set_xlabel("Alternative")
ax.set_ylabel("Segment")
ax.set_title("Logit model")  # optional: mirrors baseline

# Colorbar: use the same parameters as the baseline script
cbar = fig.colorbar(im, ax=[ax], fraction=0.046, pad=0.04)
cbar.set_label("Probability")

# Optional: add a suptitle like baseline (keeps style consistent)
plt.suptitle("Segment-level choice probabilities (multiple competitors)", fontsize=12)

plt.tight_layout()
heatmap_path = plots_dir / "Fig_MultComp_ChoiceProbabilities_Heatmap.png"
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
ax.set_title("Aggregate market shares at $x^*$ (multiple competitors)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

ms_path = plots_dir / "Fig_MultComp_MarketShares_Bar.png"
plt.savefig(ms_path, dpi=300)
plt.close(fig)
print(f"Saved: {ms_path}")

# Optional: quick console summary
print("\n====================")
print("MultComp postprocessing summary")
print("====================")
print(f"x* = [p, l2_day, l2_night] = [{p_val:.6f}, {l2d_val:.6f}, {l2n_val:.6f}]")
print(f"Total expected revenue = {rev_total:.6f}")
print(f"Market shares: Outside={share_out:.4f}, C1={share_c1:.4f}, C2={share_c2:.4f}")
print("Done.")

