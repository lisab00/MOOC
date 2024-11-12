import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from scipy.stats import multivariate_normal as mvn


n = 2  # dimension of the samples
N = 5000  # number of desired samples in the chain
x0 = np.zeros(n)  # starting point
dens = mvn(mean=np.zeros(n), cov=np.eye(n)).pdf  # density to sample from
prop_var = 2.38**2  # variance of proposal


def proposal():
    return rnd.multivariate_normal(mean=np.zeros(n), cov=prop_var*np.eye(n))


samples = np.empty((N, n))  # storage for generated samples
samples[0, :] = x0  # first element in the chain is x0

x = x0  # initialization
dens_x = dens(x)
accptd = 0  # number of accepted proposals

for j in range(1, N):
    eps = proposal()
    x_ = x + eps
    dens_x_ = dens(x_)

    accpt_prob = np.min([1, dens_x_/dens_x])

    if accpt_prob >= rnd.random():
        # accept
        x = x_
        dens_x = dens_x_
        accptd += 1

    samples[j, :] = x

    if j % 100 == 0:
        # Print acceptance rate every 100th step
        print("Acceptance rate: %f" % (accptd/j))

np.savetxt('samples.txt', samples)

plt.figure()
plt.plot(range(1, N+1), samples[:, 0])  # plot first component of all chain elements
plt.plot(range(1, N+1), samples[:, 1])  # plot first component of all chain elements
plt.tight_layout()
plt.show()
