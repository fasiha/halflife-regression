import numpy as np
import scipy.stats as stats
import scipy.special as special


def prob(k, n, t, x, w):
    h = 2**(x @ w)
    p = 2**(-t / h)
    return stats.binom.pmf(k, n, p)


def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -special.betaln(1 + n - k, 1 + k) - np.log(n + 1)


def probJacobian(k, n, t, x, w, widx):
    h = 2**(x @ w)
    p = 2**(-t / h)
    pmf = stats.binom.pmf(k, n, p)
    return pmf * x[widx] * np.log(2)**2 * t / (1 - p) * (k - n * p) / h


log2 = np.log(2)
loglog2Times2 = 2 * np.log(log2)


def probJacobianAccurate(k, n, t, x, w, widx):
    logh = x @ w * log2
    logp = -t / h * log2
    log1MinusP = np.log(-np.expm1(logp))
    logpmf = binomln(n, k) + k * logp + (n - k) * log1MinusP
    logretbase = (logpmf + loglog2Times2 + np.log(t) - logh - log1MinusP +
                  np.log(x[widx]))
    return np.exp(logretbase) * k - np.exp(logretbase + logp + np.log(n))


def testAccuracy():
    probJacobian(0, 1, 0.001597, np.array([16.822604, 12.041595, 1]),
                 np.array([1.5386902, 1.60667677, 1.69713884]), 0)
    probJacobianAccurate(0, 1, 0.001597, np.array([16.822604, 12.041595, 1]),
                         np.array([1.5386902, 1.60667677, 1.69713884]), 0)
    sumProbJac(0, 1, 0.001597, np.array([16.822604, 12.041595, 1]),
               np.array([1.5386902, 1.60667677, 1.69713884]))


def sumProbJac(k, n, t, x, w):
    h = 2**(x @ w)
    p = 2**(-t / h)
    pmf = stats.binom.pmf(k, n, p)
    jacBase = pmf * np.log(2)**2 * t / (1 - p) * (k - n * p) / h
    return (-np.sum(pmf), -(np.atleast_1d(jacBase) @ x))


def sumProbJacDf(w, x, df):
    h = 2**(x @ w)
    p = 2**(-df.t / h)
    pmf = stats.binom.pmf(df.k, df.n, p)
    # if not np.all(np.isfinite(pmf)):
    #     print('non finite pmf')
    #     pdb.set_trace()
    jacBase = pmf * np.log(2)**2 * df.t / (1 - p) * (df.k - df.n * p) / h
    # if not np.all(np.isfinite(jacBase)):
    #     print('non finite jacBase')
    #     pdb.set_trace()
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

Ndata = round(len(fulldata) * .9)
data = fulldata[:Ndata]
test = fulldata[Ndata:]


# https://github.com/benbo/adagrad/blob/master/adagrad.py
def adaGrad(weights,
            df,
            x,
            stepsize=1e-2,
            fudge_factor=1e-6,
            max_it=1000,
            minibatchsize=250,
            verbose=True):
    ld = len(data)
    gti = np.zeros_like(weights)

    for t in range(max_it):
        # https://stackoverflow.com/a/34879805/500207
        sd = df.sample(minibatchsize)
        sx = x[sd.index, :]
        val, grad = sumProbJacDf(weights, sx, sd)
        gti += grad**2
        adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
        weights -= stepsize * adjusted_grad
        if verbose:
            # prob = objective(weights, df)
            print(
                "# Iteration {}, weights={}, |grad|^2={:.1e}, Δ={:.1e}".format(
                    t, weights, np.sum(grad**2),
                    np.sqrt(np.sum(adjusted_grad**2))))
    return weights


# np.seterr(all='raise')

init = np.zeros(3)
X = np.c_[data.sqrtright, data.sqrtwrong, np.ones(len(data))]
# w = adaGrad(init, data, X, stepsize=1., max_it=20_000, minibatchsize=1000)


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
     sumProbJac(data.k, data.n, data.t, X, np.array([-.3, .01, 4.4])),
     sumProbJacDf([-.3, .01, 4.4], X, data)]
