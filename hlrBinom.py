import numpy as np
import scipy.stats as stats
import scipy.special as special


def prob(k, n, t, x, w):
    h = 2**(x @ w)
    p = 2**(-t / h)
    return stats.binom.pmf(k, n, p)


def probJacobian(k, n, t, x, w, widx):
    h = 2**(x @ w)
    p = 2**(-t / h)
    return special.binom(n, k) * x[widx] * np.log(2)**2 * t * p**k * (1 - p)**(
        n - k - 1) * (k - n * p) / h


import numdifftools as nd
k = 2
n = 3
t = 1.1
x = np.array([1, 2, 3.])
w1 = -.1
w2 = -.3
w0 = -.2
print([
    nd.Derivative(lambda w: prob(k, n, t, x, np.array([w1, w, w0])))(w2),
    probJacobian(k, n, t, x, np.array([w1, w2, w0]), 1)
])
