import numpy as np
import sympy as sp

N = 2  # number of contracts
S = 3  # number of customer segments
D = 2  # time slots (day/night)

tmin, tmax = 0.0, 40.0
beta = 0.05 
gamma = 3 

# Consumption in kWh
Es_original = np.array([
    [10, 120],
    [130, 10],
    [50, 50]
])

# Production costs
prod_cost_lambda1 = sp.Matrix([4, 2])
prod_cost_lambda2 = sp.Matrix([5, 3])

outside1 = np.array([5, 3])
#outside2 = np.array([8, 3])
#outside3 = np.array([4, 3])
competitors = [outside1]
#, outside2, outside3] # Variable fees (lambda_comp)

# We assume a fixed monthly fee (e.g., 5.0) for each competitor.
# These values are constants and are NOT being optimized.
fixed_fee_comp = 0.0 
fixed_fees_comp_list = [fixed_fee_comp]
#, fixed_fee_comp, fixed_fee_comp] 

# You must keep the variable fees in sympy for the dot product later.
competitors_sym=[sp.Matrix(c) for c in competitors]


f1, f2 = sp.symbols("f1 f2", real=True)
l11, l12, l21, l22 = sp.symbols("l11 l12 l21 l22", real=True)
variables = (f1, f2, l11, l12, l21, l22)

lambda1 = sp.Matrix([l11, l12])
lambda2 = sp.Matrix([l21, l22])

# This def provides the minimal competitor (outside option), which represents the reservation bill
def reservation_bill(E): 
    
    # E is the consumption vector for a segment E_s
    
    # Calculate the total bill for each competitor: (Fixed Fee) + (Variable Fees * Consumption)
    comp_costs = []
    for i in range(len(competitors_sym)):
        # Variable cost: E.dot(lambda_comp)
        variable_cost = E.dot(competitors_sym[i])
        
        # Total cost: Fixed Fee + Variable Cost
        total_cost = fixed_fees_comp_list[i] + variable_cost
        comp_costs.append(total_cost)
        
    return sp.Min(*comp_costs)

tmin, tmax = 0.0, 40.0

def shifting_cons(E, shift):
    E_day, E_night = E
    E_day_new = (1 - shift) * E_day
    E_night_new = E_night + shift * E_day
    return np.array([E_day_new, E_night_new])

# Apply shift
shift = 0.25
Es = np.array([shifting_cons(E, shift) for E in Es_original])

def symbolic_profit(beta, w):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        R_s = reservation_bill(E_s)

        bill1 = f1 + E_s.dot(lambda1)
        bill2 = f2 + E_s.dot(lambda2)

        #prod_cost1 = E_s.dot(prod_cost_lambda1)
        #prod_cost2 = E_s.dot(prod_cost_lambda2)

        #cost_company_prod = [prod_cost1, prod_cost2]
        rev_company = [bill1, bill2]

        # disutilities relative to reservation bill
        DU_1 = bill1 - R_s
        DU_2 = bill2 - R_s
        DU_Comp = 0

        disutilities = [DU_Comp, DU_1, DU_2] # 0 is the competitor's relative utility
        
        # Stabilization
        max_utility = sp.Max(DU_Comp, DU_1, DU_2) # Is used for stabilizing the problem below
        
        exp_term = [sp.exp(-beta * (u - max_utility)) for u in disutilities]
        denom = sum(exp_term)
        probs = [e / denom for e in exp_term]

        # Expected revenue
        total += w[s] * sum(rev_company[i] * probs[i+1] for i in range(N)) #- cost_company_prod[i] when dealing with profit
    return total

w_basic = [1, 1, 1]
profit_sym = symbolic_profit(beta, w_basic)
U_sym = -profit_sym  
gradU_sym = [sp.diff(U_sym, v) for v in variables]

# Change to numpy functions using lamdify
U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")


def U(x):
    #vals = np.clip(x, tmin, tmax) #Here we clip the variables in between tmin and tmax, to have positive values
    return float(U_func(*x))

def dU(x):
    #vals = np.clip(x, tmin, tmax)
    grad_vals = np.array(gradU_func(*x), dtype=float).flatten()
    return grad_vals