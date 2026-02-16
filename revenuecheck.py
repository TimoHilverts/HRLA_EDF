
import numpy as np

import BasicModel as base                 # U(x)
import BasicModel_Scaled as scaled        # SCALE, to_x, to_z, U(z)

np.set_printoptions(precision=12, suppress=True)

# --------------------------------------------------
# ORIGINAL VECTOR from the SCALED run (this is z*)
# --------------------------------------------------
zstar = np.array([1.0629827,  0.9750389,  0.68260916], dtype=float)

# --------------------------------------------------
# Convert z* -> x* and compute revenue in x-space
# --------------------------------------------------
xstar = scaled.to_x(zstar)

U_x = base.U_x(xstar)
revenue_x = -U_x

# --------------------------------------------------
# Compute revenue via scaled model directly
# (should match revenue_x)
# --------------------------------------------------
U_z = scaled.U_z(zstar)
revenue_z = -U_z

# --------------------------------------------------
# Also convert back: x* -> z* (sanity check)
# --------------------------------------------------
z_back = scaled.to_z(xstar)

print("SCALE:")
print(scaled.SCALE)

print("\nOriginal scaled vector z* (from HRLA):")
print(zstar)

print("\nConverted unscaled vector x* = SCALE ⊙ z*:")
print(xstar)

print("\nBack-converted z_back = x* / SCALE (should equal z*):")
print(z_back)
print(f"||z_back - z*||_inf = {np.max(np.abs(z_back - zstar)):.3e}")

print("\nRevenue checks (should match up to floating-point error):")
print(f"Revenue via -base.U(x*)     = {revenue_x:.12f}")
print(f"Revenue via -scaled.U(z*)   = {revenue_z:.12f}")
print(f"|rev_x - rev_z|             = {abs(revenue_x - revenue_z):.3e}")
