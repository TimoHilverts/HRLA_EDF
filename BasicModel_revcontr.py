import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output folder
plots_dir = Path("output/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# --- Revenue data in natural segment order (1–10) ---
segments = np.arange(1, 11)

revenue_contribution = np.array([
    104.258107,
    173.465686,
    82.091679,
    262.800566,
    60.376621,
    135.366713,
    105.797755,
    118.444693,
    56.710193,
    172.984699
], dtype=float)

# --- Plot ---
plt.figure(figsize=(8, 4.5))
plt.bar(segments, revenue_contribution)

plt.xlabel("Segment")
plt.ylabel("Expected revenue contribution (€)")
plt.title("Revenue contribution by segment (baseline model)")
plt.xticks(segments)
plt.grid(True, axis="y")

plt.tight_layout()
plt.savefig(plots_dir / "Fig_Baseline_RevenueBySegment.png", dpi=300)
plt.close()

print("Saved: Fig_Baseline_RevenueBySegment.png")
