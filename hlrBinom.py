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
    pmf = stats.binom.pmf(k, n, p)
    return pmf * x[widx] * np.log(2)**2 * t / (1 - p) * (k - n * p) / h


def sumProbJac(k, n, t, x, w):
    h = 2**(x @ w)
    p = 2**(-t / h)
    pmf = stats.binom.pmf(k, n, p)
    jacBase = pmf * np.log(2)**2 * t / (1 - p) * (k - n * p) / h
    return (-np.sum(pmf), -(jacBase @ x))


import pandas as pd
fulldata = pd.read_csv("features.csv")
fulldata['n'] = fulldata.session_seen
fulldata['k'] = fulldata.session_correct
fulldata['t'] = fulldata.delta / (60 * 60 * 24)  # convert time delta to days
fulldata['sqrtright'] = np.sqrt(1 + fulldata.history_correct)
fulldata['sqrtwrong'] = np.sqrt(1 + (fulldata.history_seen -
                                     fulldata.history_correct))
fulldata['obsp'] = fulldata.session_correct / fulldata.session_seen

Ndata = 1_000
data = fulldata[:Ndata]
# test = fulldata[Ndata:]


def testJac():
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

    def obj(k, n, t, x, w):
        h = 2**(x @ w)
        p = 2**(-t / h)
        pmf = stats.binom.pmf(k, n, p)
        return -np.sum(pmf)

    X = np.c_[data.sqrtright, data.sqrtwrong, np.ones(len(data))]
    [[
        nd.Derivative(lambda w: obj(data.k, data.n, data.t, X,
                                    np.array([w, .01, 4.4])))(-.3),
        nd.Derivative(lambda w: obj(data.k, data.n, data.t, X,
                                    np.array([-.3, w, 4.4])))(.01),
        nd.Derivative(lambda w: obj(data.k, data.n, data.t, X,
                                    np.array([-.3, .01, w])))(4.4)
    ],
     sumProbJac(data.k, data.n, data.t, X, np.array([-.3, .01, 4.4]))]
