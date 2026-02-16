import numpy as np

import ConShift as model  # provides SCALE, to_x, U_x, U_z (as defined in your file)

np.set_printoptions(precision=12, suppress=True)

# --------------------------------------------------
# ORIGINAL VECTOR from the SCALED run (this is z*)
# Order in ConShift.py:
# (f1, p, f2, l2_day, l2_night)
# --------------------------------------------------
zstar = np.array([0.52820839, 1.03699322, 0.711057,   0.9334883,  1.03094615], dtype=float)  # <-- REPLACE with your Habrok z* (length 5)

# --------------------------------------------------
# Convert z* -> x* and compute revenue in x-space
# --------------------------------------------------
xstar = model.to_x(zstar)

U_x = model.U_x(xstar)
revenue_x = -U_x

# --------------------------------------------------
# Compute revenue via scaled model directly
# (should match revenue_x if U_z is consistent)
# --------------------------------------------------
U_z = model.U_z(zstar)
revenue_z = -U_z

# --------------------------------------------------
# Also convert back: x* -> z* (sanity check)
# (ConShift.py does not define to_z by default, so we do it here)
# --------------------------------------------------
SCALE = np.asarray(model.SCALE, dtype=float)
z_back = np.asarray(xstar, dtype=float) / SCALE

print("SCALE:")
print(SCALE)

print("\nOriginal scaled vector z* (from HRLA):")
print(zstar)

print("\nConverted unscaled vector x* = SCALE ⊙ z*:")
print(xstar)

print("\nBack-converted z_back = x* / SCALE (should equal z*):")
print(z_back)
print(f"||z_back - z*||_inf = {np.max(np.abs(z_back - zstar)):.3e}")

print("\nRevenue checks (should match up to floating-point error):")
print(f"Revenue via -U_x(x*)     = {revenue_x:.12f}")
print(f"Revenue via -U_z(z*)     = {revenue_z:.12f}")
print(f"|rev_x - rev_z|          = {abs(revenue_x - revenue_z):.3e}")
