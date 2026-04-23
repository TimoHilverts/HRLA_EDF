import numpy as np

# ============================================================
# Rastrigin function
# ============================================================
def U_rastrigin(x: np.ndarray) -> float:
    d = len(x)
    return float(d + np.sum(x**2 - np.cos(2 * np.pi * x)))

# ============================================================
# ESS check
# ============================================================
d = 10
np.random.seed(42)

# Search space matching paper's initialization N(3*1_d, 10*I_d)
# covers roughly mean=3, std=sqrt(10)~3.16, so [-5, 5] is reasonable
z_test_points = np.random.uniform(-5, 5, size=(1000, d))
U_vals = np.array([U_rastrigin(z) for z in z_test_points])

print(f"U_rastrigin statistics over search space:")
print(f"  Mean : {U_vals.mean():.2f}")
print(f"  Std  : {U_vals.std():.2f}")
print(f"  Min  : {U_vals.min():.2f}")
print(f"  Max  : {U_vals.max():.2f}")
print()

print("ESS check:")
for a_min in [0.1, 0.01, 0.001, 0.0001]:
    weights = np.exp(-a_min * U_vals)
    weights /= weights.sum()
    ess = 1 / np.sum(weights**2)
    print(f"  a_min={a_min:.4f} | Max weight: {weights.max():.4f} | ESS: {ess:.1f}")