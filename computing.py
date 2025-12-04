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
        K=500, 
        h=0.01, 
        title=title, 
        U=model.U, 
        dU=model.dU, 
        initial=initial
    )
    
    # This generates the file, with different a values
    samples_filename = algorithm.generate_samples(As=[0.1, 1, 5, 10, 100], sim_annealing=True)
    
    print(f"Optimization finished.")

    #Optimal x* = [17.3643073  -0.47725905 -0.0796222  11.92164123  4.98177168  4.06638457] for best a = 5

    #note that I used K = 500, since 15000 was too much to do it locally.
    
    # Best value for a=0.1: [31.5090317  34.3794968  -0.09871693 11.64686149  8.81392038 18.11460619]
    # Best value for a=1: [20.01170654 26.74498958  0.37504354 10.80055873 11.73042645  8.84210742]
    # Best value for a=5: [17.3643073  -0.47725905 -0.0796222  11.92164123  4.98177168  4.06638457]
    # Best value for a=10: [14.03775699  5.21488433  1.49809849  8.38482751 12.91239263  2.81129973]
    # Best value for a=100: [-2.33520143  6.20728039  0.99259572  9.92980756  7.65162235  3.8659051 ]