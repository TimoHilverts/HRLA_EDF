import numpy as np
import matplotlib.pyplot as plt
import os
from PostProcessing import PostProcessor
import fullmodel as model # Importing from file 1 to get U(x)

filename = "temp_output/data/HRLA_basicmodel_d6_1764862760.4485607.pickle" 

print(f"Analyzing {filename}...")

postprocessor = PostProcessor(filename)

postprocessor.compute_tables([10, 100, 3000, 15000], 1, "best")

# Get Best Candidates
bests = postprocessor.get_best(measured=[10,100,3000,15000], dpi=10)

# Recalculate revenues using the shared model.U function
revenues_best = [-model.U(np.array(x)) for x in bests]

best_idx = int(np.argmax(revenues_best)) #gives highest revenue, and then as an integer
best_a = postprocessor.As[best_idx]
x_opt = np.array(bests[best_idx])

print(f"Best a = {best_a}")
print(f"Optimal x* = {x_opt}")

var_names = ["f1", "f2", "l11", "l12", "l21", "l22"]
tvals = np.linspace(-10, model.tmax, 400)

output_dir = "output/plots"
os.makedirs(output_dir, exist_ok=True)

print("Generating 1D cut plots...")

for i, name in enumerate(var_names):
    revenues = []
    
    # Calculate the curve
    for t in tvals:
        x_temp = x_opt.copy()
        x_temp[i] = t
        # We use model.U here to get the physics
        revenues.append(-model.U(x_temp))

    revenues = np.array(revenues)

    # Find max along this specific 1D cut
    idx_max = np.argmax(revenues)
    t_max = tvals[idx_max]
    R_max = revenues[idx_max]

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(tvals, revenues, color="black", label=f"Revenue along 1D cut of {name}")
    
    # HRLA optimum (Red)
    plt.axvline(x=x_opt[i], color="red", linestyle="--",
                label=f"HRLA optimum {name}* = {x_opt[i]:.2f}")
    
    # True 1D optimum (Green)
    plt.axvline(x=t_max, color="green", linestyle="--",
                label=f"1D cut optimum = {t_max:.2f}")

    plt.title(f"1D cuts of revenue vs {name} (best a = {best_a})")
    plt.xlabel(name)
    plt.ylabel("Revenue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save as plot
    plot_path = f"{output_dir}/1Dcuts_tmin-10_{name}_best_a_{best_a}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved: {plot_path}")

print("Done.")