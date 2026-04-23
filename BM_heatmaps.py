import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import BasicModel as model

# --------------------------------------------------
# Hardcoded optimal solution (baseline)
# --------------------------------------------------
x_star = np.array([0.332865, 0.348804, 0.178977], dtype=float)

# --------------------------------------------------
# Model primitives
# --------------------------------------------------
Es = np.asarray(model.Es_used, dtype=float)
beta = float(model.beta)
w = np.asarray(model.w_uniform, dtype=float)
comp = np.asarray(model.competitor, dtype=float)

S = Es.shape[0]

# --------------------------------------------------
# Compute bills
# Alternatives: [Outside, Contract 1 (Flat), Contract 2 (ToU)]
# --------------------------------------------------
p_val, l2d_val, l2n_val = x_star

bill_out = Es[:, 0] * comp[0] + Es[:, 1] * comp[1]
bill_c1  = Es[:, 0] * p_val   + Es[:, 1] * p_val
bill_c2  = Es[:, 0] * l2d_val + Es[:, 1] * l2n_val

B = np.column_stack([bill_out, bill_c1, bill_c2])

# --------------------------------------------------
# Logit probabilities
# --------------------------------------------------
Eexp = np.exp(-beta * B)
P = Eexp / Eexp.sum(axis=1, keepdims=True)

# --------------------------------------------------
# Deterministic allocation
# --------------------------------------------------
choice_det = np.argmin(B, axis=1)
P_det = np.zeros_like(P)
for s in range(S):
    P_det[s, choice_det[s]] = 1.0

# --------------------------------------------------
# Plot side-by-side heatmaps
# --------------------------------------------------
plots_dir = Path("output/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(10, 4.5))

# Logit heatmap
ax1 = fig.add_subplot(1, 2, 1)
im1 = ax1.imshow(P, aspect="auto")
ax1.set_title("Logit model")
ax1.set_xlabel("Alternative")
ax1.set_ylabel("Segment")
ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(["Outside", "C1", "C2"])
ax1.set_yticks(np.arange(S))
ax1.set_yticklabels(np.arange(1, S + 1))

# Deterministic heatmap
ax2 = fig.add_subplot(1, 2, 2)
im2 = ax2.imshow(P_det, aspect="auto")
ax2.set_title("Deterministic limit")
ax2.set_xlabel("Alternative")
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(["Outside", "C1", "C2"])
ax2.set_yticks(np.arange(S))
ax2.set_yticklabels(np.arange(1, S + 1))

# Shared colorbar
cbar = fig.colorbar(im1, ax=[ax1, ax2], fraction=0.046, pad=0.04)
cbar.set_label("Probability")

plt.suptitle("Comparison of logit and deterministic segment allocation", fontsize=12)
plt.tight_layout()
plt.savefig(plots_dir / "Fig_Baseline_Logit_vs_Deterministic_Heatmaps.png", dpi=300)
plt.close()

print("Saved: Fig_Baseline_Logit_vs_Deterministic_Heatmaps.png")
