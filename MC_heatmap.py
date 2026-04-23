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
