'''
setup file
- create the two different bayesian models to be analyzed
- create the rwmh algorithm code
'''

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvnorm
from pathlib import Path

main_dir = Path(r'C:\Users\lihel\Documents\MOOC\exercise_1')

class bayes_model:

    def __init__(self, data, noise_level, n):
        self.data = data
        self.q = data['Q']
        self.Gamma = np.diag(noise_level * self.q) #Gamma depends on noise level and is set from task to task
        self.n = n
        self.m0 = np.zeros(n)
        self.Sigma0 = 10 * np.eye(n)

    def prior(self, x):
        return mvnorm(mean=self.m0, cov=self.Sigma0).pdf(x)

    def likelihood(self, x):
        temp = self.q - self.A @ x
        #return np.exp(-1/2 * temp.T @ self.Gamma @ temp) #different matrix norm
        Gamma_inv_sqrt = np.diag(1. / np.sqrt(np.diag(self.Gamma))) # norm from bayesian inversian lecture video (block 1)
        return np.exp(-1/2 * np.linalg.norm(Gamma_inv_sqrt @ temp)**2.)

    def posterior(self, x):
        return self.prior(x) * self.likelihood(x)

    def formula_post_mean_var(self):
        mean = self.m0 + self.Sigma0 @ self.A.T @ np.linalg.inv(self.Gamma + self.A @ self.Sigma0 @ self.A.T) @ (self.q - self.A @ self.m0)
        cov = self.Sigma0 - self.Sigma0 @ self.A.T @ np.linalg.inv(self.Gamma + self.A @ self.Sigma0 @ self.A.T) @ self.A @ self.Sigma0
        return (mean, cov)


class model1(bayes_model):

    def __init__(self, data, noise_level, n):
        super().__init__(data, noise_level, n)
        self.A = np.column_stack((np.ones(len(data)), np.log(data['H'].values)))
        

class model2(bayes_model):

    def __init__(self, data, noise_level, n):
        super().__init__(data, noise_level, n)
        self.A = np.column_stack((np.ones(len(data)), data['H'].values, data['H'].values**2))
        print(self.A)
    

class rwmh:

    def __init__(self, model, prop_var, x0, N):
        self.sigma2 = prop_var
        self.model = model
        self.n = model.n # dimension of the samples
        self.x0 = x0
        self.N = N

    def proposal(self):
        return rnd.multivariate_normal(mean=np.zeros(self.n), cov=self.sigma2*np.eye(self.n))
        # return mvnorm(mean=np.zeros(self.n), cov=self.sigma2*np.eye(self.n)).rvs() # scipy is slower
    
    def execute(self):

        samples = np.empty((self.N, self.n)) # storage for generated samples
        samples[0, :] = self.x0 # frist element in the chain is x0

        x = self.x0 # initialization
        dens_x = self.model.posterior(x) # density to be sampled from
        accptd = 0 # number of accepted proposals

        for j in range(1, self.N):
            xp = x + self.proposal() # x proposed sample
            dens_xp = self.model.posterior(xp)

            acc_prob = np.min([1, dens_xp / dens_x])

            if acc_prob >= rnd.random(): # accept xp
                x = xp
                dens_x = dens_xp
                accptd += 1

            samples[j, :] = x

            if j % 1000 == 0:
                # Print acceptance rate every 100th step
                print("Acceptance rate: %f" % (accptd/j))

        np.savetxt(main_dir / "outputs" / "model_2" / "samples" / "samples.txt", samples, delimiter=',')
        plt.figure()
        for i in np.arange(self.n):
            plt.plot(range(1, self.N+1), samples[:, i])  # plot first component of all chain elements
        plt.tight_layout()
        plt.savefig('plot.png')
        plt.show()
