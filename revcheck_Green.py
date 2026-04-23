import numpy as np

import GreenComp as model  # provides SCALE, to_x, U_x, U_z (as defined in your file)

np.set_printoptions(precision=12, suppress=True)

# --------------------------------------------------
# ORIGINAL VECTOR from the SCALED run (this is z*)
# Order in GreenComp.py:
# (f1,p1,f2,p2,f3,l3_day,l3_night,f4,l4_day,l4_night)
# --------------------------------------------------
zstar = np.array([0.79705935, 0.99959628, 0.78293557, 1.00224319, 0.96490036, 0.93897612,
 0.83128338, 0.95870656, 0.94240011, 0.82098973], dtype=float)  # <-- REPLACE with your Habrok z*

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
# (GreenComp.py does not define to_z by default, so we do it here)
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
