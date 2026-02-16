# postprocessing_BasicModel_HARDCODED_deterministic.py
#
# Standalone deterministic (large-beta) evaluation at the hardcoded baseline optimum x*.
# - No pickles.
# - Computes:
#     * deterministic segment choices (argmin bill)
#     * deterministic total retailer revenue
#     * deterministic revenue per contract
#     * number of segments choosing each alternative
# - (Optional) saves a LaTeX table in the "nice" grouped-header style.
#
# Assumes BasicModel.py defines: Es_used (S,2), competitor (2,), w_uniform (S,).
# (If w_uniform is not present, we fall back to uniform weights.)

import numpy as np
import pandas as pd
from pathlib import Path

import BasicModel as model

# ----------------------------
# Hardcoded optimum (x-space)
# x = [p, l2_day, l2_night]
# ----------------------------
x_star = np.array([0.332865, 0.348804, 0.178977], dtype=float)

# ----------------------------
# Output dir
# ----------------------------
tables_dir = Path("output/tables")
tables_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Load primitives
# ----------------------------
Es = np.asarray(model.Es_used, dtype=float)  # (S,2)
comp = np.asarray(model.competitor, dtype=float).flatten()  # (2,)

S = Es.shape[0]
if Es.shape[1] != 2:
    raise ValueError(f"Expected Es_used with 2 columns (day/night), got shape {Es.shape}.")

if hasattr(model, "w_uniform"):
    w = np.asarray(model.w_uniform, dtype=float).flatten()
    if w.size != S:
        raise ValueError(f"w_uniform has length {w.size}, but Es_used has {S} segments.")
else:
    w = np.ones(S, dtype=float) / S

# ----------------------------
# Compute bills at x*
# Alternatives: 0=Outside, 1=Contract 1 (Flat), 2=Contract 2 (ToU)
# ----------------------------
p_val, l2d_val, l2n_val = map(float, x_star)

bill_out = Es[:, 0] * comp[0] + Es[:, 1] * comp[1]
bill_c1  = Es[:, 0] * p_val   + Es[:, 1] * p_val
bill_c2  = Es[:, 0] * l2d_val + Es[:, 1] * l2n_val

B = np.column_stack([bill_out, bill_c1, bill_c2])  # (S,3)

# ----------------------------
# Deterministic choices: pick lowest bill
# ----------------------------
choice_det = np.argmin(B, axis=1)  # 0,1,2

# ----------------------------
# Deterministic revenue (retailer only)
# Outside yields 0 retailer revenue.
# ----------------------------
rev_det_seg = np.zeros(S, dtype=float)
rev_det_seg[choice_det == 1] = w[choice_det == 1] * bill_c1[choice_det == 1]
rev_det_seg[choice_det == 2] = w[choice_det == 2] * bill_c2[choice_det == 2]

rev_total_det = float(rev_det_seg.sum())

rev_det_c1 = float(np.sum(w[choice_det == 1] * bill_c1[choice_det == 1]))
rev_det_c2 = float(np.sum(w[choice_det == 2] * bill_c2[choice_det == 2]))
rev_det_out = 0.0

# Segment counts (not weights)
n_out = int(np.sum(choice_det == 0))
n_c1  = int(np.sum(choice_det == 1))
n_c2  = int(np.sum(choice_det == 2))

# ----------------------------
# Print summary
# ----------------------------
print("\n====================")
print("Deterministic evaluation at x* (baseline)")
print("====================")
print(f"x* = [p, l2_day, l2_night] = [{p_val:.6f}, {l2d_val:.6f}, {l2n_val:.6f}]")
print("")
print("Segments choosing:")
print(f"  Outside option      : {n_out}")
print(f"  Contract 1 (Flat)   : {n_c1}")
print(f"  Contract 2 (ToU)    : {n_c2}")
print("")
print("Deterministic retailer revenue (€):")
print(f"  Contract 1 (Flat)   : {rev_det_c1:.6f}")
print(f"  Contract 2 (ToU)    : {rev_det_c2:.6f}")
print(f"  Outside option      : {rev_det_out:.2f}")
print(f"  TOTAL               : {rev_total_det:.6f}")
print("====================\n")

# ----------------------------
# Optional: save a LaTeX table in the exact grouped-header style you like
# (This is the same style as your screenshot table.)
# ----------------------------
beta_val = getattr(model, "beta", None)
beta_str = f"{float(beta_val):.2f}" if beta_val is not None else "NA"

table_tex = r"""
\begin{table}[ht]
\centering
\caption{Comparison between smooth (logit) and deterministic customer choice at the optimized contracts $x^*$.}
\label{tab:baseline_logit_vs_deterministic_nice}
\begin{tabular}{lccc}
\toprule
& \multicolumn{1}{c}{\textbf{Logit model}} 
& \multicolumn{2}{c}{\textbf{Deterministic model}} \\
& \multicolumn{1}{c}{$(\beta = %s)$} 
& \multicolumn{2}{c}{$(\beta \rightarrow \infty)$} \\
\cmidrule(lr){2-2} \cmidrule(lr){3-4}
& Expected revenue 
& Segments choosing 
& Revenue \\
\midrule
Outside option & -- & %d & %.2f \\
Contract 1 (Flat) & -- & %d & %.4f \\
Contract 2 (ToU)  & -- & %d & %.4f \\
\midrule
\textbf{Total company revenue} & -- &  & \textbf{%.4f} \\
\bottomrule
\end{tabular}
\end{table}
""" % (beta_str, n_out, rev_det_out, n_c1, rev_det_c1, n_c2, rev_det_c2, rev_total_det)

tex_path = tables_dir / "Table_Deterministic_Baseline_Nice.tex"
tex_path.write_text(table_tex, encoding="utf-8")
print(f"Saved LaTeX table: {tex_path}")

# ----------------------------
# Optional: save per-segment deterministic choice details (CSV + LaTeX)
# ----------------------------
labels = np.array(["Outside", "Contract 1 (Flat)", "Contract 2 (ToU)"], dtype=object)

df = pd.DataFrame({
    "Segment": np.arange(1, S + 1),
    "Bill outside": bill_out,
    "Bill contract 1": bill_c1,
    "Bill contract 2": bill_c2,
    "Chosen alternative": labels[choice_det],
    "Deterministic revenue contribution": rev_det_seg,
})

df.to_csv(tables_dir / "Deterministic_Baseline_PerSegment.csv", index=False)
df.to_latex(tables_dir / "Deterministic_Baseline_PerSegment.tex", index=False, float_format=lambda x: f"{x:.6f}")
print("Saved per-segment deterministic details (CSV + TeX).")
