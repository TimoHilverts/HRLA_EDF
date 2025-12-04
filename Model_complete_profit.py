import numpy as np
import sympy as sp
from sympy import Piecewise, Eq
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor, Comparator
import matplotlib.pyplot as plt 
import os

N = 3  #number of contracts
S = 3  #number of customer segments
D = 2  #time slots (day/night)

Es_original = np.array([ #production in kWh
    [10, 120],
    [130, 10],
    [50, 50]
])

#We make symbolic notation of the components in the contracts.
#we have fixed fees f_i, green fixed fees gf_i, variable fees l_ij, and variable green fees g_i
f1, f2 = sp.symbols("f1 f2", real=True)
l11, l12, l21, l22 = sp.symbols("l11 l12 l21 l22", real=True)
gf, g1, g2 = sp.symbols("gf g1 g2", real=True)   

lambda1=sp.Matrix([l11,l12]) #contract 1, with l11 price for day, and l12 price at night
lambda2=sp.Matrix([l21,l22]) #contract 2, with l21 price for day, and l22 price at night
lambda_green=sp.Matrix([g1, g2]) #green contract, with green prices

lambdas = [lambda1, lambda2, lambda_green]

#These represents the production costs in kWh, during day and night time
prod_cost_lambda1=sp.Matrix([4, 2])
prod_cost_lambda2=sp.Matrix([5, 3])
prod_cost_lambda_green=sp.Matrix([7, 4]) #higher production costs for green contracts

#Below, we have multiple competitors
outside1 = np.array([6, 1])
outside2 = np.array([8, 3])
outside3 = np.array([4, 3])

competitors = [outside1, outside2, outside3] #combining the competitors into one matrix

competitors_sym=[sp.Matrix(c) for c in competitors]


#This def provides the minimal competitor (outside option), which represents the reservation bill
def reservation_bill(E): 
    comp_costs=[E.dot(c) for c in competitors_sym]
    
    return sp.Min(*comp_costs)

tmin, tmax = 0.0, 40.0


def shifting_cons(E, shift):
    #E=[E_day, E_night]
    E_day, E_night = E
    E_day_new = (1-shift)*E_day
    E_night_new = E_night+shift*E_day
    
    return np.array([E_day_new, E_night_new])

shift=0.25
Es=np.array([shifting_cons(E, shift) for E in Es_original])

gamma=3 #Represents whether customers prefer green or not.
        #gamma > 0 -> customer prefers green

#Below we determine the revenue as a def using sympy. In the shifted_exp part, we subtract -max_c from the exponent part.
#Reason for this is that this stabilizes the problem. It prevents both underflow and overflow.
#e^(-beta*C) -> 0 for large C, or -> 0 for small C, such that it might become undefined in later stages, which this normalization prevents.
def symbolic_revenue(beta, w):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        R_s=reservation_bill(E_s) #provides the reservation bill, so the minimum of outsiders

        bill1=f1+E_s.dot(lambda1) #represents company cost, looking at contract 1
        bill2=f2+E_s.dot(lambda2) #represents company cost, looking at contract 2
        bill_green=gf+E_s.dot(lambda_green) #the cost for a green contract

        prod_cost1=E_s.dot(prod_cost_lambda1)
        prod_cost2=E_s.dot(prod_cost_lambda2)
        prod_cost_green=E_s.dot(prod_cost_lambda_green)

        cost_company_prod=[prod_cost1, prod_cost2, prod_cost_green] #The total production cost for the company

        rev_company=[bill1, bill2, bill_green] #represents the revenue based on the bills of the segments for the company

        cost_customer=[R_s, bill1, bill2, bill_green]

        DU_1=bill1-R_s #this is the disutility, considering cost 1, which is U =C1-Rs. Represents the difference between the 
                            # the first contract and minimum competitor, which we want to minimize of course
        DU_2=bill2-R_s

        DU_green=bill_green-R_s-gamma #here we include gamma, to include a term for incorporating green behavior. Gamma>0 -> higher green preference

        disutilities=[0, DU_1, DU_2, DU_green] #here, the utility of the competitor is 0

        max_utility=sp.Max(0, DU_1, DU_2, DU_green) #Needed for stabilizing the problem

        #max_c = sp.Max(*[-beta * c for c in cost_customer]) #here, we include the max_c term to make sure the model remains stable
        exp_term = [sp.exp(-beta * (u - max_utility)) for u in disutilities]
        #exp_term = [sp.exp(-beta * c - max_c) for c in cost_customer]
        denom = sum(exp_term)
        probs = [e / denom for e in exp_term]

        #exclude the outside option for the revenue
        total += w[s]*sum((rev_company[i]- cost_company_prod[i]) * probs[i+1] for i in range(N))
    return total

w_basic=[1, 1, 1]
w_equal=[1/3, 1/3, 1/3]
#w_diff=[w1, w2, w3]

beta = 0.05 

#define U_sym as -revenue, needed for finding the maximum revenue
revenue_sym = symbolic_revenue(beta, w_basic) #here we now include w_basic=[1, 1, 1] to check the basic case
U_sym = -revenue_sym  

variables=(f1, f2, l11, l12, l21, l22, gf, g1, g2)
gradU_sym = [sp.diff(U_sym, v) for v in variables] #this is the gradient wrt all lambdas

#reformulating U and dU as a lambdify function, converting it back to numpy notation
U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

def U(x):
    #Here, we now have x = [gf, f1, f2, g1, g2, l11, l12, l21, l22]!!
    vals = np.clip(x, tmin, tmax)
    return float(U_func(*vals))

def dU(x):
    vals = np.clip(x, tmin, tmax)
    grad_vals = np.array(gradU_func(*vals), dtype=float).flatten()
    return grad_vals


d = 9 #number of optimization variables, it tells HRLA how many coordinates lambda_i it needs to update
title = "HRLA_basicmodel_d9"

def initial():
    return np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))

algorithm = GO.HRLA(d=d, M=1, N=1, K=1500, h=0.01, title=title, U=U, dU=dU, initial=initial)
samples_filename_HRLA = algorithm.generate_samples(As=[1, 2, 3, 4, 5], sim_annealing=False)


#d is the number of optimization variables.
#It tells HRLA how many coordinates lambda_i it needs to update

#M is the number of independent runs. Each run starts from a different random initialization
#For increasing M you can average over runs (improves confidence)

#N is number of samples per iteration per run. At each iteration k,
#HRLA may draw N parallel samples X_k^n per run

#K is number of iterations. Each step evolves the stochastic diff equation
#Larger K -> more steps, better convergence (samples closer to equilibrium)
#too small K -> samples haven't yet reached stationary distribution

#h is step size, U(x) is what HRLA minimizes, dU(x) analytic gradient

#As=[...] are the inverse temperature parameters a.
#Large a -> more concentrated around minima

#sim_annealing: When False, a stays fixed per run.
#When True: a increases with iterations (can accelerate convergence when a is large)

postprocessor = PostProcessor(samples_filename_HRLA)
postprocessor.compute_tables([10, 500, 1500], 1, "best") #This provides the tables with a and K vals, together with optimal revenue

#plot the empirical probability distributions for all four components l_ij
#postprocessor.plot_empirical_probabilities(layout="22", tols=[1,2,3,4], dpi=1, running=True)

#With this call (the get_best call), we find for every value for a the best choices for the parameters l_ij, which achieve
#the optimal revenue.
bests = postprocessor.get_best(measured=[10,500,1500], dpi=1)