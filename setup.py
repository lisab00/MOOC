'''
setup file
- create the two different bayesian models to be analyzed
- create the rwmh algorithm code
'''

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvnorm

class bayes_model:

    m0 = np.array([0.,0.])
    Sigma0 = np.array([[10.,0.], [0., 10.]])

    def __init__(self, data, noise_level):
        self.data = data
        self.q = data['Q']
        self.Gamma = np.diag(noise_level * np.log(self.q)) #Gamma depends on noise level and is set from task to task

    def prior(self, x):
        return mvnorm(mean=self.m0, cov=self.Sigma0).pdf(x)

    def likelihood(self, x):
        temp = np.log(self.q) - self.A @ x
        return np.exp(-1/2 * temp.T @ self.Gamma @ temp) # TODO check matrix norm

    def posterior(self, x):
        return self.prior(x) * self.likelihood(x)

    def formula_post_mean_var(self):
        mean = self.m0 + self.Sigma0 @ self.A.T @ np.linalg.inv(self.Gamma + self.A @ self.Sigma0 @ self.A.T) @ (self.q- self.A @ self.m0)
        cov = self.Sigma0 - self.Sigma0 @ self.A.T @ np.linalg.inv(self.Gamma + self.A @ self.Sigma0 @ self.A.T) @ self.A @ self.Sigma0
        return (mean, cov)


class model1(bayes_model):

    def __init__(self, data, noise_level):
        super().__init__(data, noise_level)
        self.A = np.column_stack((np.ones(len(data)), np.log(data['H'].values)))
        

class model2(bayes_model):

    def __init__(self, data, noise_level):
        super().__init__(data, noise_level)
        self.A = 0 # TODO
    

class rwmh:

    n = 2 # dimension of the samples

    def __init__(self, model, prop_var, x0, N):
        self.sigma2 = prop_var
        self.model = model
        self.x0 = x0
        self.N = N

    def proposal(self):
        #return rnd.multivariate_normal(mean=np.zeros(self.n), cov=self.sigma2*np.eye(self.n))
        return mvnorm(mean=np.zeros(self.n), cov=self.sigma2*np.eye(self.n)).rvs()
    
    def execute(self):

        samples = np.empty((self.N, self.n)) # storage for generated samples
        samples[0, :] = self.x0 # frist element in the chain is x0

        x = self.x0 # initialization
        dens_x = self.model.posterior(x)
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

            if j % 100 == 0:
                # Print acceptance rate every 100th step
                print("Acceptance rate: %f" % (accptd/j))

        np.savetxt('samples.txt', samples, delimiter=',')
        plt.figure()
        plt.plot(range(1, self.N+1), samples[:, 0])  # plot first component of all chain elements
        plt.plot(range(1, self.N+1), samples[:, 1])  # plot first component of all chain elements
        plt.tight_layout()
        plt.savefig('plot.png')
        plt.show()
