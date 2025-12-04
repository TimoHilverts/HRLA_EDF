import numpy as np
import matplotlib.pyplot as plt
import os
from PostProcessing import PostProcessor
import fullmodel as model # Importing from file 1 to get U(x)

# Here we display the filename
filename = "temp_output/data/HRLA_basicmodel_d6_1764862760.4485607.pickle" 

postprocessor = PostProcessor(filename)

# Compute the tables
postprocessor.compute_tables([10, 100, 3000, 15000], 1, "best")

# Get the best candidates, by using the get_best function from PostProcessor
bests = postprocessor.get_best(measured=[10, 100, 3000, 15000], dpi=10)

# Define the specific a we want to plot
TARGET_A = 5
try:
    # Here, postprocesor.As is the list of a's we want to consider
    target_a_idx = postprocessor.As.index(TARGET_A) 
except ValueError:
    print(f"Error: Target inverse temperature a={TARGET_A} not found in the run data ({postprocessor.As}).")
    exit()

# We set the optimal parameters based on the chosen index
best_a = TARGET_A
x_opt = np.array(bests[target_a_idx]) 

# Recalculate and print the revenue for this chosen set of parameters
max_revenue = -model.U(x_opt)

print(f"Plotting 1D cuts for specific a = {best_a}")
print(f"Corresponding optimal x* = {x_opt}")
print(f"Revenue at this x* = {max_revenue:.4f}")

var_names = ["f1", "f2", "l11", "l12", "l21", "l22"]
tvals = np.linspace(-40, model.tmax, 400)

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

    plt.title(f"1D dpi1 cuts of revenue vs {name} (best a = {TARGET_A})")
    plt.xlabel(name)
    plt.ylabel("Revenue")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plots
    plot_path = f"{output_dir}/1Dcuts_a1tmin-40_{name}_best_a_{TARGET_A}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved: {plot_path}")

print("Done.")