import pickle
import builtins

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Python2 -> Python3 builtins mapping
        if module == "__builtin__":
            module = "builtins"
        return super().find_class(module, name)

def load_compat_pickle(path):
    with open(path, "rb") as f:
        # encoding is ignored for protocol >= 3, but helps for Py2 pickles
        return CompatUnpickler(f).load()

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import GreenComp as model  # must match the run used to create the pickle


# ============================================================
# USER SETTINGS
# ============================================================

PICKLE_FILE = "temp_output/data/FullModel_N4_FixedFees_Shift_Green_1770990259.856996.pickle"  # <-- change
# If your run used As=[10] only, A_INDEX=0 is fine.
A_INDEX = 0

# If you want checkpoints instead of every iteration:
MEASURED = [100, 500, 1000, 3000, 5000, 7500, 10000, 20000, 30000, 40000, 50000]

OUT_DIR = Path("output/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper: find arrays that look like trajectories
# ============================================================

def _collect_arrays(obj, found):
    """Recursively collect numpy arrays contained in a pickle structure."""
    if isinstance(obj, np.ndarray):
        found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_arrays(v, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_arrays(v, found)

def find_candidate_trajectories(data, d_expected=None):
    """
    Return a list of arrays that look like trajectories.
    Typical shapes:
      (K, d)
      (M, K, d)
      (len(As), K, d)
      (len(As), M, K, d)
    """
    arrays = []
    _collect_arrays(data, arrays)

    cands = []
    for a in arrays:
        if a.ndim >= 2:
            # heuristic: last dimension might be d
            if d_expected is None or (a.shape[-1] == d_expected):
                cands.append(a)

    # prefer larger arrays
    cands.sort(key=lambda x: x.size, reverse=True)
    return cands


# ============================================================
# Main
# ============================================================

def main():
    data = load_compat_pickle(PICKLE_FILE)


    d = len(getattr(model, "SCALE"))
    cands = find_candidate_trajectories(data, d_expected=d)

    if not cands:
        raise RuntimeError(
            "Could not find any numpy arrays in the pickle that look like trajectories.\n"
            "If you paste the output of `type(data)` and the keys (if dict), I can tailor the parser."
        )

    # Print a short summary so you see what was found
    print("Found candidate arrays (largest first):")
    for i, a in enumerate(cands[:8]):
        print(f"  cand[{i}] shape={a.shape}, dtype={a.dtype}")

    traj = cands[0]  # take the largest candidate by default

    # Now reduce traj to a list of z-paths with shape (K,d)
    # Handle a few common layouts.
    z_paths = []

    if traj.ndim == 2 and traj.shape[1] == d:
        # (K,d)
        z_paths = [traj]

    elif traj.ndim == 3 and traj.shape[-1] == d:
        # could be (M,K,d) or (A,K,d)
        # decide based on which axis is "K" (usually the middle one is K for (M,K,d))
        if traj.shape[1] > traj.shape[0]:
            # assume (M,K,d)
            z_paths = [traj[m] for m in range(traj.shape[0])]
        else:
            # assume (A,K,d)
            z_paths = [traj[A_INDEX]]

    elif traj.ndim == 4 and traj.shape[-1] == d:
        # could be (A,M,K,d) or (M,A,K,d)
        # assume first axis is A
        z_block = traj[A_INDEX]  # shape (M,K,d)
        z_paths = [z_block[m] for m in range(z_block.shape[0])]

    else:
        raise RuntimeError(
            f"Unsupported trajectory array shape: {traj.shape}\n"
            "Paste this shape and I will adapt the extractor."
        )

    # Evaluate revenue along all paths, take best-so-far across paths at each iteration
    # Revenue = -U_x(SCALE*z)
    # We compute best-so-far over iterations and over paths.
    Kmax = min(p.shape[0] for p in z_paths)
    print(f"Using {len(z_paths)} path(s), Kmax={Kmax}, d={d}")

    best_so_far = np.full(Kmax, -np.inf, dtype=float)

    for m, z in enumerate(z_paths):
        # truncate to common length
        z = np.asarray(z[:Kmax], dtype=float)

        # evaluate revenue along iterations for this path
        rev = np.empty(Kmax, dtype=float)
        for k in range(Kmax):
            x = model.to_x(z[k])
            rev[k] = float(-model.U_x(x))

        # best-so-far within this path
        rev_best_path = np.maximum.accumulate(rev)

        # aggregate best-so-far across paths
        best_so_far = np.maximum(best_so_far, rev_best_path)

        print(f"  done path {m+1}/{len(z_paths)}: final best={best_so_far[-1]:.6f}")

    # Prepare measured points
    measured = [k for k in MEASURED if k <= Kmax]
    if not measured:
        measured = list(range(1, Kmax + 1))

    Ks = np.array(measured, dtype=int)
    R = best_so_far[Ks - 1]  # if iteration index starts at 1 in your reporting

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(Ks, R, marker="o")
    plt.xlabel("Iterations K")
    plt.ylabel("Best revenue found")
    plt.grid(True)

    j = int(np.nanargmax(R))
    plt.title(f"GreenComp: Best revenue vs K (best at K={Ks[j]}, rev={R[j]:.2f})")

    out_path = OUT_DIR / "GreenComp_revenue_vs_K.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\nSaved plot: {out_path}")
    print(f"Best checkpoint: K={Ks[j]}, revenue={R[j]:.12f}")


if __name__ == "__main__":
    main()
