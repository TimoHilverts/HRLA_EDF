import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sympy as sp
import GlobalOptimizationHRLA as GO
from PostProcessing import PostProcessor

st.title("The Optimal Contract Finder")

uploaded_file = st.file_uploader("Upload consumption matrix E", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    Es = df.values
    st.dataframe(df)

else:
    st.warning("Please upload a consumption matrix to continue.")
    st.stop()  # stops the script here

S, D = Es.shape
N_contracts = 2

st.sidebar.header("Parameters")
beta = st.sidebar.number_input("β (rationality)", min_value=0.001, value=0.05)
K = st.sidebar.number_input("K (iterations)", min_value=10, max_value=2000, value=300)
N_samples = st.sidebar.number_input("N (samples per HRLA iteration)", min_value=1, max_value=20, value=1)
a_values = st.sidebar.text_input("Enter a list of a-values (comma separated)", "1,3,5")
a_list = [float(x) for x in a_values.split(",")]

# Define symbolic lambdas (two contracts, two time slots)
l11, l12, l21, l22 = sp.symbols("l11 l12 l21 l22", real=True)

outside = sp.Matrix([6, 1])
lambda1 = sp.Matrix([l11, l12])
lambda2 = sp.Matrix([l21, l22])

lambdas = [outside, lambda1, lambda2]

def symbolic_revenue(beta):
    total = 0
    for s in range(S):
        E_s = sp.Matrix(Es[s])
        costs = [E_s.dot(lam) for lam in lambdas]
        max_c = sp.Max(*[-beta*c for c in costs])
        shifted_exp = [sp.exp(-beta*c - max_c) for c in costs]
        denom = sum(shifted_exp)
        probs = [e/denom for e in shifted_exp]
        total += sum(costs[i] * probs[i] for i in range(1, N_contracts+1))
    return total

revenue_sym = symbolic_revenue(beta)
U_sym = -revenue_sym
variables = (l11, l12, l21, l22)
gradU_sym = [sp.diff(U_sym, v) for v in variables]

U_func = sp.lambdify(variables, U_sym, "numpy")
gradU_func = sp.lambdify(variables, gradU_sym, "numpy")

tmin, tmax = 0, 40

def U(x):
    vals = np.clip(x, tmin, tmax)
    return float(U_func(*vals))

def dU(x):
    vals = np.clip(x, tmin, tmax)
    grad_vals = np.array(gradU_func(*vals), dtype=float).flatten()
    return grad_vals

d=4
title="HRLA_basicmodel_d4"

def initial():
    return np.random.multivariate_normal(3*np.ones(d), 10*np.eye(d))

if st.button("Find best contract prices!"):

    # Let user enter multiple values of 'a'


    algorithm = GO.HRLA(
        d=d, M=1, N=N_samples, K=K, h=0.01,
        title=title, U=U, dU=dU, initial=initial
    )

    # Run HRLA for all provided a-values
    samples_filename_HRLA = algorithm.generate_samples(As=a_list, sim_annealing=False)

    # Extract results
    postprocessor = PostProcessor(samples_filename_HRLA)
    bests = postprocessor.get_best(measured=[K], dpi=1)

    # Compute revenues for each a
    revenues = []
    for params in bests:
        p = np.array(params, dtype=float).ravel()
        rev = -U(p)
        revenues.append(rev)

    # Find the index of the best revenue
    idx_best = np.argmax(revenues)
    best_params = np.array(bests[idx_best], dtype=float).ravel()
    best_rev = revenues[idx_best]
    best_a = a_list[idx_best]

    # Show results
    st.success("Completed")
    st.write(f"Best a-value: {best_a}")
    st.write(f"Optimal parameters (l11, l12, l21, l22): {best_params}")
    st.write(f"Maximum revenue: {best_rev:.2f}")
