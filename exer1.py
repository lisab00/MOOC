from pathlib import Path
import numpy as np
import pandas as pd
import random as rnd

main_dir = Path(r'C:\Users\lihel\Documents\MOOC')

# load data
df_data = pd.read_csv('rc_Q_V.txt', sep=',')
print(df_data)

# create error array
error = np.zeros(len(df_data))
print(error)

# compute mean and cov matrix of posterior distribution
m0 = np.array([0.,0.])
q = df_data['Q']
Sigma0 = np.array([[10.,0.], [0., 10.]])
Gamma = np.diag(0.03 * np.log(q))
A = np.zeros((len(df_data), 2))
for i in range(len(df_data)): 
    A[i] = [1, np.log(df_data['H'].iloc[i])]

post_mean = m0 + Sigma0 @ A.T @ np.linalg.inv(Gamma + A @ Sigma0 @ A.T) @ (q- A @ m0)
post_cov = Sigma0 - Sigma0 @ A.T @ np.linalg.inv(Gamma + A @ Sigma0 @ A.T) @ A @ Sigma0
print(f"the posterior mean by formula is {post_mean}")
print(f"the posterior coveriance by formula is {post_cov}")

def prior(m, sigma):
    return rnd.normal(m, sigma)

def likelihood(q, x, A, Gamma):
    temp = np.log(q) - A @ x
    return np.exp(-1/2 * temp.T @ Gamma @ temp)

def prod_prior_likeli(prior, likeli):
    return prior * likeli
