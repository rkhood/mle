'''
fit line by maximum likelihood estimation
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

np.random.seed(123)
plt.style.use('ggplot')

def make_data(m=5, b=2, n=20):

        x = np.random.uniform(0,10,n)
        y = m*x + b
        u = np.random.normal(0, 5, n)
        y += np.random.rand(n)*u
        theta = [m, b]

        return theta, x, y, u


def log_likelihood(theta, x, y, u):

       m, b = theta
       n = len(x)
       f = lambda x: m*x + b
       log_l = -0.5 * np.sum(((y-f(x))/u)**2 + n * np.log(u**2))

       return log_l


def mle(x, y, u, log_l):

        nll = lambda *args: -1 * log_l(*args)
        result = optimize.minimize(nll, [1, 1], args=(x, y, u),
                        method='Nelder-Mead')

        return result.x

        
# an example
m, b = 5, 3
theta, x, y, u = make_data(m, b)
mle_theta = mle(x, y, u, log_likelihood)

plt.errorbar(x, y, u, fmt='.')
plt.plot(x, mle_theta[0] * x + mle_theta[1])
plt.show()
