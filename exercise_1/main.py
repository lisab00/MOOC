from pathlib import Path
import pandas as pd
import numpy as np
import random as rd
import matplotlib as plt
import setup

def main():

    # set seed
    np.random.seed(123)

    # load data for model 1 (here we need log(Q)) # for model 2 load data newly
    main_dir = Path(r'C:\Users\lihel\Documents\MOOC\exercise_1')
    data_log = pd.read_csv(main_dir / 'rc_Q_V.txt', sep=',')
    data_log['Q'] = np.log(data_log['Q'])

    ### perform tasks for model1

    # create model1 instance
    model1 = setup.model1(data_log, noise_level=0.03)

    # compute mean and covariance of posterior by formula
    formula_results = model1.formula_post_mean_var()
    print(f"the posterior mean by formula is {formula_results[0]}")
    print(f"the posterior coveriance by formula is {formula_results[1]}")

    # execute the rwmh algorithm
    rwmh = setup.rwmh(model=model1, prop_var=2.38 ** 2, n=2, x0=np.zeros(2), N=20000)
    rwmh.execute()

    # read rwmh outputs
    df_output_mod1 = pd.read_csv('samples.txt', sep=',', header=None)
    print(df_output_mod1.head(15))
    
    # drop burin samples after visually examining results
    val = input("drop rows including index: ")
    df_output_mod1.drop(index=np.arange(int(val)+1), inplace=True)

    # plot parameter densities and auto correlations, plot the rating curve

    # compute rwmh mean and variance
    rwmh_results = (np.mean(df_output_mod1, axis=0), np.cov(df_output_mod1, rowvar=False))
    print(f"the posterior mean by rwmh algo is {rwmh_results[0]}")
    print(f"the posterior coveriance by rwmh algo is {rwmh_results[1]}")
    

if __name__ == '__main__':
    main()