# --- Analyze_HRLA_Results.py ---
import pickle
import numpy as np

# 👇 Paste here the path printed by BasicModel_HRLA.py
samples_filename_HRLA = "temp_output/data/HRLA_basic_1762709350.0761428.pickle"

# Compatibility loader (handles Python2 pickles)
def load_pickle_python2_compat(path):
    with open(path, "rb") as f:
        raw = f.read()
    raw = raw.replace(b"__builtin__", b"builtins")
    return pickle.loads(raw, encoding="latin1")

# Load pickle file
data = load_pickle_python2_compat(samples_filename_HRLA)
print("✅ Successfully loaded HRLA results.")
print("Keys:", data.keys())

# Extract and analyse
X = np.array(data["X"]).reshape(-1)
U_vals = np.array(data["U"]).reshape(-1)
idx_opt = int(np.argmin(U_vals))
t_opt = float(X[idx_opt])
R_opt = -U_vals[idx_opt]

print(f"\n⚙️  HRLA optimum found:")
print(f"   t* ≈ {t_opt:.4f}")
print(f"   Maximum revenue ≈ {R_opt:.4f}")
