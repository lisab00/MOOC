from pathlib import Path
import pandas as pd
import numpy as np
import random as rd
import matplotlib as plt
import setup

def main():

    # set seed and path
    np.random.seed(123)
    main_dir = Path(r'C:\Users\lihel\Documents\MOOC\exercise_1')

    ### perform tasks for model1
    nl = 0.6           # noise level
    pvar= 3**2       # proposal variance

    # load data for model 1 (here we need log(Q)) # for model 2 load data newly
    data_log = pd.read_csv(main_dir / 'rc_Q_V.txt', sep=',')
    data_log['Q'] = np.log(data_log['Q'])

    # create model1 instance
    model1 = setup.model1(data_log, noise_level=nl, n=2)

    # compute mean and covariance of posterior by formula
    formula_results = model1.formula_post_mean_var()
    print(f"the posterior mean by formula is {formula_results[0]}")
    print(f"the posterior covariance by formula is {formula_results[1]}")

    # execute the rwmh algorithm
    rwmh = setup.rwmh(model=model1, prop_var=pvar, x0=np.zeros(2), N=30000)
    rwmh.execute()

    # read rwmh outputs
    df_output_mod1 = pd.read_csv(main_dir / "outputs" / "model_1" / "samples" / "samples.txt", sep=',', header=None)
    print(df_output_mod1.head(15))
    
    # drop burin samples after visually examining results
    val = input("drop rows including index: ")
    df_output_mod1.drop(index=np.arange(int(val)+1), inplace=True)
    np.savetxt(main_dir / "outputs" / "model_1" / "samples" / "samples.txt", df_output_mod1, delimiter=',')

    # compute rwmh mean and variance
    rwmh_results = (np.mean(df_output_mod1, axis=0), np.cov(df_output_mod1, rowvar=False))
    print(f"the posterior mean by rwmh algo is {rwmh_results[0]}")
    print(f"the posterior covariance by rwmh algo is {rwmh_results[1]}")
    

    ### perform tasks for model2
    nl = 20000               # noise level
    pvar= 2.38**2        # proposal variance

    # load data for model 1 (here we need log(Q)) # for model 2 load data newly
    data = pd.read_csv(main_dir / 'rc_Q_V.txt', sep=',')

    # create model2 instance
    model2 = setup.model2(data, noise_level=nl, n=3)

    # compute mean and covariance of posterior by formula
    formula_results = model2.formula_post_mean_var()
    print(f"the posterior mean by formula is {formula_results[0]}")
    print(f"the posterior covariance by formula is {formula_results[1]}")
    
    # execute the rwmh algorithm
    rwmh = setup.rwmh(model=model2, prop_var=pvar, x0=np.zeros(3), N=5000)
    rwmh.execute()

    # read rwmh outputs
    df_output_mod2 = pd.read_csv(main_dir / "outputs" / "model_2" / "samples" / "samples.txt", sep=',', header=None)
    print(df_output_mod2.head(15))
    
    # drop burin samples after visually examining results
    val = input("drop rows including index: ")
    df_output_mod2.drop(index=np.arange(int(val)+1), inplace=True)
    np.savetxt(main_dir / "outputs" / "model_2" / "samples" / "samples.txt", df_output_mod2, delimiter=',')

    # compute rwmh mean and variance
    rwmh_results = (np.mean(df_output_mod2, axis=0), np.cov(df_output_mod2, rowvar=False))
    print(f"the posterior mean by rwmh algo is {rwmh_results[0]}")
    print(f"the posterior covariance by rwmh algo is {rwmh_results[1]}")
    

if __name__ == '__main__':
    main()