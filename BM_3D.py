import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import BasicModel as model

output_dir = Path("output/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# --- choose economically meaningful bounds ---
# Replace these with your true feasible bounds
bounds = {
    "p": (0.0, 0.6),
    "l2_day": (0.0, 0.6),
    "l2_night": (0.0, 0.6),
}

N = 20000  # number of random samples (increase if cheap)
rng = np.random.default_rng(0)

P = rng.uniform(*bounds["p"], size=N)
D = rng.uniform(*bounds["l2_day"], size=N)
Nn = rng.uniform(*bounds["l2_night"], size=N)

X = np.column_stack([P, D, Nn])

rev = np.empty(N)
for i in range(N):
    rev[i] = -model.U_x(X[i])

mask = np.isfinite(rev)
X = X[mask]
rev = rev[mask]

# keep only top fraction for clarity
q = np.quantile(rev, 0.90)  # show top 10%
sel = rev >= q

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(X[sel,0], X[sel,1], X[sel,2], c=rev[sel], s=6)

ax.set_xlabel("p")
ax.set_ylabel("l2_day")
ax.set_zlabel("l2_night")
fig.colorbar(sc, ax=ax, shrink=0.7, label="Revenue")
ax.set_title("3D top-revenue point cloud (top 10%)")

out = output_dir / "Fig_3D_PointCloud_Top10pct.png"
plt.tight_layout()
plt.savefig(out, dpi=300)
plt.close(fig)
print("Saved:", out)
