import numpy as np
import GlobalOptimizationHRLA as GO

# IMPORTANT:
# - BasicModel_Scaled.U expects z and internally evaluates base.U(SCALE*z)
# - BasicModel_Scaled.dU expects z and returns the z-gradient
import BasicModel_Scaled as model_scaled

title = "BasicModel_scaled"
d = 6  # (p1, p2, l31, l32, l41, l42) in z-space


def initial():
    """
    Initial point in z-space.
    z ≈ [1,1,1,1,1,1] corresponds to x ≈ SCALE, i.e., competitor-like baseline tariffs.
    """
    z0 = np.ones(d, dtype=float)

    # Small noise in z-space (dimensionless, comparable across coordinates)
    z0 += np.random.normal(loc=0.0, scale=0.05, size=d)

    return z0


if __name__ == "__main__":
    print("Starting Optimization (scaled variables via BasicModel_Scaled wrapper)")

    algorithm = GO.HRLA(
        d=d,
        M=1,
        N=1,
        K=1000,
        h=0.0001,              # single step size
        title=title,
        U=model_scaled.U,      # <-- U(z)
        dU=model_scaled.dU,    # <-- dU(z)
        initial=initial
    )

    samples_filename = algorithm.generate_samples(
        As=[5, 50, 100, 500, 1000, 3000, 5000],
        sim_annealing=True
    )

    print(f"Optimization finished. Samples saved to: {samples_filename}")
    print("NOTE: saved samples are z-vectors. Convert to real tariffs via x = model_scaled.to_x(z).")
