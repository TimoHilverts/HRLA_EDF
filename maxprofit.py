import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy as sp
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

st.title("The Optimal Contract Finder")
uploaded_file = st.file_uploader("The consumption Es of the customer segment", type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    Es=df.values
    st.dataframe(df)

st.sidebar.header("Parameters")
beta = st.sidebar.number_input("β (rationality)", min_value=0.001, value=0.05)
K = st.sidebar.number_input("K (iterations)", min_value=10, max_value=2000, value=300)
N_samples = st.sidebar.number_input("N (samples per iteration)", min_value=1, max_value=20, value=1)
a_val = st.sidebar.number_input("a (inverse temperature)", min_value=1, max_value=100, value=10)

S, D=Es.shape

l11, l12, l21, l22 = sp.symbols("l11 l12 l21 l22", real=True)

outside = sp.Matrix([6,1])
lambda1=sp.Matrix([l11,l12]) #contract 1, with l11 price for day, and l12 price at night
lambda2=sp.Matrix([l21,l22]) #contract 2, with l21 price for day, and l22 price at night

lambdas = [outside, lambda1, lambda2]

#Below we determine the revenue as a def using sympy. In the shifted_exp part, we subtract -max_c from the exponent part.
#Reason for this is that this stabilizes the problem. It prevents both underflow and overflow.
#e^(-beta*C) -> 0 for large C, or -> 0 for small C, such that it might become undefined in later stages, which this normalization prevents.
def symbolic_revenue(beta):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        costs = [E_s.dot(lam) for lam in lambdas] 
        max_c = sp.Max(*[-beta * c for c in costs]) #here, we include the max_c term to make sure the model remains stable
        shifted_exp = [sp.exp(-beta * c - max_c) for c in costs]
        denom = sum(shifted_exp)
        probs = [e / denom for e in shifted_exp]

        #exclude the outside option for the revenue
        total += sum(costs[i] * probs[i] for i in range(1, N+1))
    return total

beta = 0.05 

#define U_sym as -revenue, needed for finding the maximum revenue
revenue_sym = symbolic_revenue(beta)
U_sym = -revenue_sym  

variables = (l11, l12, l21, l22)
gradU_sym = [sp.diff(U_sym, v) for v in variables] #this is the gradient wrt all lambdas

#reformulating U and dU as a lambdify function, converting it back to numpy notation
U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

def U(x):
    #Here, we now have x = [l11, l12, l21, l22]!!
    vals = np.clip(x, tmin, tmax)
    return float(U_func(*vals))

def dU(x):
    vals = np.clip(x, tmin, tmax)
    grad_vals = np.array(gradU_func(*vals), dtype=float).flatten()
    return grad_vals

d = 4 #number of optimization variables, it tells HRLA how many coordinates lambda_i it needs to update
title = "HRLA_basicmodel_d4"

def initial():
    return np.random.multivariate_normal(np.zeros(d) + 3, 10 * np.eye(d))


if st.button("Optimize!"):

    algorithm = GO.HRLA(d=d, M=1, N=N, K=K, h=0.01, title=title, U=U, dU=dU, initial=initial)
    samples_filename_HRLA = algorithm.generate_samples(As=[a], sim_annealing=False)

    postprocessor = PostProcessor(samples_filename_HRLA)

    #plot the empirical probability distributions for all four components l_ij
    #postprocessor.plot_empirical_probabilities(layout="22", tols=[1,2,3,4], dpi=1, running=True)

    #With this call (the get_best call), we find for every value for a the best choices for the parameters l_ij, which achieve
    #the optimal revenue.
    bests = postprocessor.get_best(measured[K], dpi=1)

    l11_opt, l12_opt, l21_opt, l22_opt=bests
    best_rev=-U(bests)

    st.success("Optimization complete")

    st.write(f"The max revenue is {best_rev:.2f}")








    