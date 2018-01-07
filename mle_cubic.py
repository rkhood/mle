import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

np.random.seed(123)
plt.style.use('ggplot')

def make_data(a0=2, a1=-3, a2=-3, a3=3, n=20):

        x = np.sort(np.random.uniform(-2, 2,n))
        y = a0 + a1*x + a2*x*x + a3*x*x*x
        u = np.random.normal(0, 1, n)
        y += np.random.rand(n)*u

        return x, y, u


def log_likelihood(theta, x, y, u):

       n = len(x)
       d = np.vstack([np.ones(n), x, x*x, x*x*x])
       f = lambda x: theta.dot(d)
       log_l = -0.5 * np.sum(((y-f(x))/u)**2 + n * np.log(u**2))

       return log_l


def mle(x, y, u, log_l):

        nll = lambda *args: -1 * log_l(*args)
        result = optimize.minimize(nll, np.array([1, 1, 1, 1]), args=(x, y, u),
                        method='Nelder-Mead')

        return result.x

        
# an example
x, y, u = make_data()
mle_theta = mle(x, y, u, log_likelihood)

plt.errorbar(x, y, u, fmt='.')
xp = np.linspace(-2, 2, 100)
plt.plot(xp, mle_theta.dot(np.vstack([np.ones(len(xp)), xp, xp*xp, xp*xp*xp])))
plt.show()
