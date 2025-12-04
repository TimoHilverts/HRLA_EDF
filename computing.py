import numpy as np
import GlobalOptimizationHRLA as GO
import fullmodel as model # Importing from file fullmodel

title = "HRLA_d6"
d = 6  # number of variables
def initial():
    return np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

#This runs the optimization
if __name__ == "__main__":
    print("Starting Optimization")
    
    # We pass the U and dU from the model module
    algorithm = GO.HRLA(
        d=d, 
        M=1, 
        N=1, 
        K=15000, 
        h=0.01, 
        title=title, 
        U=model.U, 
        dU=model.dU, 
        initial=initial
    )
    
    # This generates the file
    samples_filename = algorithm.generate_samples(As=[0.1, 1, 5, 10, 100], sim_annealing=True)
    
    print(f"Optimization finished.")