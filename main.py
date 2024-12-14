from pathlib import Path
import pandas as pd
import numpy as np
import setup

def main():

    # load data
    main_dir = Path(r'C:\Users\lihel\Documents\MOOC')
    data = pd.read_csv('rc_Q_V.txt', sep=',')

    ### perform tasks for model1

    # create model1 instance
    model1 = setup.model1(data, noise_level=0.03)

    # compute mean and covariance of posterior by formula
    formula_results = model1.formula_post_mean_var()
    print(f"the posterior mean by formula is {formula_results[0]}")
    print(f"the posterior coveriance by formula is {formula_results[1]}")

    # execute the rwmh algorithm
    rwmh = setup.rwmh(model=model1, prop_var=2.38**2, x0=np.zeros(2), N=5000)
    rwmh.execute()
    

if __name__ == '__main__':
    main()