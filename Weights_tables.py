# postprocessing_Weights_HARDCODED_tables_figures.py
#
# Same as before, but:
# (1) Table 4 (revenue contribution by segment) is kept in NORMAL segment order (1..S),
#     and Figure 3 uses that same normal order.
# (2) Table 1 (optimal contract menu) ALSO displays the expected total revenue.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import Weights as model


# ----------------------------
# Choose how you want to input the solution
# ----------------------------
USE_Z_SPACE = True  # True: provide z* and use model.SCALE; False: provide x* directly

z_habrok = np.array([1.01625335, 0.93533078, 0.66067948], dtype=float)  # only if USE_Z_SPACE=True
x_habrok = np.array([0.332865, 0.348804, 0.178977], dtype=float)  # <-- replace by Weights optimum if different


# ----------------------------
# Names + output dirs
# ----------------------------
var_names = ["p", "l2_day", "l2_night"]

plots_dir = Path("output/plots")
tables_dir = Path("output/tables")
plots_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Resolve x*
# ============================================================
SCALE = np.asarray(getattr(model, "SCALE", None), dtype=float) if hasattr(model, "SCALE") else None

if USE_Z_SPACE:
    if SCALE is None:
        raise ValueError("USE_Z_SPACE=True but Weights.py has no SCALE.")
    z_star = np.asarray(z_habrok, dtype=float).flatten()
    x_star = SCALE * z_star
else:
    x_star = np.asarray(x_habrok, dtype=float).flatten()
    z_star = None

p_val, l2d_val, l2n_val = map(float, x_star)


# ============================================================
# Model primitives
# ============================================================
Es = np.asarray(model.Es_used, dtype=float)                 # (S,2)
beta = float(model.beta)
w = np.asarray(model.w_data, dtype=float).flatten()         # (S,)
comp = np.asarray(model.competitor, dtype=float).flatten()  # (2,)

S = Es.shape[0]
if Es.shape[1] != 2:
    raise ValueError(f"Expected Es_used with 2 columns (day/night), got shape {Es.shape}.")
if w.shape[0] != S:
    raise ValueError(f"Expected w_data with length S={S}, got shape {w.shape}.")

# normalize weights defensively
w_sum = float(np.sum(w))
if w_sum <= 0:
    raise ValueError("w_data must sum to a positive value.")
w = w / w_sum


# ============================================================
# Bills and probabilities at x*
# Alternatives: [Outside, Contract 1 (Flat), Contract 2 (ToU)]
# ============================================================
bill_out = Es[:, 0] * comp[0] + Es[:, 1] * comp[1]
bill_c1  = Es[:, 0] * p_val   + Es[:, 1] * p_val
bill_c2  = Es[:, 0] * l2d_val + Es[:, 1] * l2n_val

B = np.column_stack([bill_out, bill_c1, bill_c2])  # (S,3)

Eexp = np.exp(-beta * B)
P = Eexp / Eexp.sum(axis=1, keepdims=True)          # (S,3)

# Expected revenue per segment (outside yields 0)
rev_seg = w * (bill_c1 * P[:, 1] + bill_c2 * P[:, 2])
rev_total = float(rev_seg.sum())

# Contract-level expected revenue + market shares
rev_contract = np.array([
    float(np.sum(w * bill_c1 * P[:, 1])),
    float(np.sum(w * bill_c2 * P[:, 2])),
], dtype=float)

share_out = float(np.sum(w * P[:, 0]))
share_c1  = float(np.sum(w * P[:, 1]))
share_c2  = float(np.sum(w * P[:, 2]))

# Deterministic evaluation (each segment picks lowest bill)
choice_det = np.argmin(B, axis=1)  # 0=outside,1=c1,2=c2
rev_det_seg = np.zeros(S, dtype=float)
rev_det_seg[choice_det == 1] = w[choice_det == 1] * bill_c1[choice_det == 1]
rev_det_seg[choice_det == 2] = w[choice_det == 2] * bill_c2[choice_det == 2]
rev_total_det = float(rev_det_seg.sum())

# Entropy for assignment summary
eps = 1e-16
entropy = -np.sum(P * np.log(P + eps), axis=1)


# ============================================================
# Helper: write LaTeX table files
# ============================================================
def save_df(df: pd.DataFrame, stem: str, floatfmt="%.6f"):
    csv_path = tables_dir / f"{stem}.csv"
    tex_path = tables_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    df.to_latex(tex_path, index=False, float_format=lambda x: floatfmt % x)
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


# ============================================================
# Table 1 — Optimal contract menu + expected total revenue
# ============================================================
p4, l2d4, l2n4 = (round(p_val, 4), round(l2d_val, 4), round(l2n_val, 4))

table1_tex = r"""
\begin{table}[ht]
\centering
\caption{Optimal tariff parameters under the weighted-segments model. The last row reports the resulting expected total revenue.}
\label{tab:weights_contract_menu}
\begin{tabular}{lcc}
\toprule
 & Contract 1 (Flat) & Contract 2 (ToU) \\
\midrule
Peak (€/kWh)      & %.4f & %.4f \\
Off-peak (€/kWh)  & %.4f & %.4f \\
\midrule
Expected total revenue & \multicolumn{2}{c}{%.6f} \\
\bottomrule
\end{tabular}
\end{table}
""" % (p4, l2d4, p4, l2n4, rev_total)

(table_path := tables_dir / "Table1_Weights_OptimalContractMenu.tex").write_text(table1_tex, encoding="utf-8")
print(f"Saved: {table_path}")

df_t1 = pd.DataFrame(
    {"Variable": var_names, "Optimal value (x*)": [p_val, l2d_val, l2n_val]}
)
save_df(df_t1, "Table1_Weights_OptimalContractMenu_raw", floatfmt="%.6f")


# ============================================================
# Table 2 — Segment-level choice probabilities
# ============================================================
df_t2 = pd.DataFrame(P, columns=["Outside", "Contract 1", "Contract 2"])
df_t2.insert(0, "Segment", np.arange(1, S + 1))
df_t2.insert(1, "Weight", w)
df_t2["Row sum"] = df_t2[["Outside", "Contract 1", "Contract 2"]].sum(axis=1)
save_df(df_t2, "Table2_Weights_SegmentChoiceProbabilities", floatfmt="%.6f")


# ============================================================
# Figure 1 — Heatmap of choice probabilities
# ============================================================
fig = plt.figure(figsize=(7.2, 4.2))
ax = fig.add_subplot(111)
im = ax.imshow(P, aspect="auto")
ax.set_yticks(np.arange(S))
ax.set_yticklabels([f"{i}" for i in range(1, S + 1)])
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Outside", "C1", "C2"])
ax.set_xlabel("Alternative")
ax.set_ylabel("Segment")
ax.set_title("Choice probabilities at the optimal contract menu (weighted model)")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Probability")
plt.tight_layout()
fig_path = plots_dir / "Fig1_Weights_ChoiceProbabilities_Heatmap.png"
plt.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Saved: {fig_path}")


# ============================================================
# Figure 2 — Outside-option take-up by segment
# ============================================================
fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_subplot(111)
ax.bar(np.arange(1, S + 1), P[:, 0])
ax.set_xlabel("Segment")
ax.set_ylabel("Outside-option probability")
ax.set_title("Outside-option take-up by segment (weighted model, at $x^*$)")
ax.set_xticks(np.arange(1, S + 1))
ax.grid(True, axis="y")
plt.tight_layout()
fig_path = plots_dir / "Fig2_Weights_OutsideOption_Takeup.png"
plt.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Saved: {fig_path}")


# ============================================================
# Table 3 — Segment assignment summary (argmax + entropy)
# ============================================================
argmax_alt = np.argmax(P, axis=1)
labels = np.array(["Outside", "Contract 1", "Contract 2"])
df_t3 = pd.DataFrame({
    "Segment": np.arange(1, S + 1),
    "Weight": w,
    "Most likely choice": labels[argmax_alt],
    "Max probability": np.max(P, axis=1),
    "Entropy": entropy,
})
save_df(df_t3, "Table3_Weights_SegmentAssignmentSummary", floatfmt="%.6f")


# ============================================================
# Table 4 — Revenue contribution by segment (NORMAL order)
# ============================================================
df_t4 = pd.DataFrame({
    "Segment": np.arange(1, S + 1),
    "Weight": w,
    "Revenue contribution": rev_seg,
    "Revenue share": rev_seg / (rev_total if rev_total > 0 else 1.0),
})
# IMPORTANT: do NOT sort; keep normal order
save_df(df_t4, "Table4_Weights_RevenueBySegment", floatfmt="%.6f")


# ============================================================
# Figure 3 — Revenue contribution per segment (NORMAL order)
# ============================================================
fig = plt.figure(figsize=(7.2, 3.6))
ax = fig.add_subplot(111)
ax.bar(np.arange(1, S + 1), df_t4["Revenue contribution"].values)
ax.set_xlabel("Segment")
ax.set_ylabel("Expected revenue contribution")
ax.set_title("Revenue contribution per segment (weighted model)")
ax.set_xticks(np.arange(1, S + 1))
ax.grid(True, axis="y")
plt.tight_layout()
fig_path = plots_dir / "Fig3_Weights_RevenueBySegment.png"
plt.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Saved: {fig_path}")


# ============================================================
# Table 5 — Revenue decomposition by contract (+ shares)
# ============================================================
df_t5 = pd.DataFrame({
    "Metric": [
        "Total expected revenue",
        "Expected revenue: Contract 1 (flat)",
        "Expected revenue: Contract 2 (ToU)",
        "Expected market share: Outside",
        "Expected market share: Contract 1",
        "Expected market share: Contract 2",
    ],
    "Value": [
        rev_total,
        rev_contract[0],
        rev_contract[1],
        share_out,
        share_c1,
        share_c2,
    ],
})
save_df(df_t5, "Table5_Weights_RevenueByContractAndShares", floatfmt="%.6f")


# ============================================================
# Figure 4 — Bills per segment (outside vs C1 vs C2)
# ============================================================
x = np.arange(1, S + 1)
width = 0.25

fig = plt.figure(figsize=(9.0, 4.0))
ax = fig.add_subplot(111)
ax.bar(x - width, bill_out, width=width, label="Outside")
ax.bar(x,         bill_c1, width=width, label="Contract 1 (flat)")
ax.bar(x + width, bill_c2, width=width, label="Contract 2 (ToU)")
ax.set_xlabel("Segment")
ax.set_ylabel("Annual bill")
ax.set_title("Annual electricity bills by segment at the optimal contract menu (weighted model)")
ax.set_xticks(x)
ax.grid(True, axis="y")
ax.legend()
plt.tight_layout()
fig_path = plots_dir / "Fig4_Weights_BillsBySegment.png"
plt.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Saved: {fig_path}")


# ============================================================
# Table 6 — Logit vs deterministic revenue
# ============================================================
df_t6 = pd.DataFrame({
    "Model": ["Logit (weighted)", "Deterministic limit (prices fixed at $x^*$)"],
    "Total expected revenue": [rev_total, rev_total_det],
    "Difference": [np.nan, rev_total_det - rev_total],
    "Relative difference": [np.nan, (rev_total_det - rev_total) / (rev_total if rev_total != 0 else np.nan)],
})
save_df(df_t6, "Table6_Weights_LogitVsDeterministic", floatfmt="%.6f")


# ============================================================
# Figure 5 — Deterministic segment allocation
# ============================================================
fig = plt.figure(figsize=(7.2, 3.2))
ax = fig.add_subplot(111)

pos_map = {"Outside": 0, "Contract 1": 1, "Contract 2": 2}
chosen_labels = labels[choice_det]
y = np.array([pos_map[l] for l in chosen_labels], dtype=int)

ax.scatter(np.arange(1, S + 1), y)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Outside", "Contract 1", "Contract 2"])
ax.set_xticks(np.arange(1, S + 1))
ax.set_xlabel("Segment")
ax.set_ylabel("Chosen alternative")
ax.set_title("Deterministic segment allocation at the weighted-model optimum $x^*$")
ax.grid(True, axis="y")
plt.tight_layout()
fig_path = plots_dir / "Fig5_Weights_DeterministicAllocation.png"
plt.savefig(fig_path, dpi=300)
plt.close(fig)
print(f"Saved: {fig_path}")

print("\nDone. (1D cuts are excluded by design.)")
