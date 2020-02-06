import scipy.optimize as opt
from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.special import psi


def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def logp(k, alpha, beta, delta, n):
    i = np.arange(n - k + 1.0)
    mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
    signs = (-1.0)**i
    return binomln(n, k) - betaln(alpha, beta) + logsumexp(
        mags, b=signs, return_sign=False)


def logpJacobianLinearAlpha(k, alpha, beta, delta, n, feature):
    i = np.arange(n - k + 1.0)
    d = delta * (i + k)
    p = psi(alpha + beta) + psi(alpha + d) - psi(alpha) - psi(alpha + beta + d)
    return p * feature


def logpJacobianLinearBeta(k, alpha, beta, delta, n, feature):
    i = np.arange(n - k + 1.0)
    d = delta * (i + k)
    p = psi(alpha + beta) - psi(alpha + beta + d)
    return p * feature


def objective(weights, datadict):
    alphas = np.clip(datadict['X'] @ weights[:2] + weights[2],
                     a_min=1.,
                     a_max=None)
    betas = np.clip(datadict['X'] @ weights[3:5] + weights[5],
                    a_min=1.,
                    a_max=None)
    deltas = datadict['deltas']
    ns = datadict['ns']
    ks = datadict['ks']
    return -sum(map(logp, ks, alphas, betas, deltas, ns))


million = pd.read_csv("pymcmill.csv")  # features.csv for full
million['sqrtright'] = np.sqrt(1 + million.history_correct)
million['sqrtwrong'] = np.sqrt(1 + (million.history_seen -
                                    million.history_correct))
million['days'] = million.delta / (3600 * 24)
data = million[:1000]

datadict = dict(X=np.array([data.sqrtright.values, data.sqrtwrong.values]).T,
                deltas=data.days.values,
                ns=data.session_seen.values,
                ks=data.session_correct.values)

init = np.array([7., -30, 48, 1, 2, 3])
init = np.array([1., 0., 1., 0., 1., 1.])
sol = opt.minimize(lambda x: objective(x, datadict), init)

import numdifftools as nd


def f(value,
      valueIdx,
      weights,
      datadict,
      jacobian=False,
      dataidx=0,
      verbose=False):
    weights = weights.copy()
    weights[valueIdx] = value
    alpha = np.clip(datadict['X'][dataidx] @ weights[:2] + weights[2],
                    a_min=1.,
                    a_max=None)
    beta = np.clip(datadict['X'][dataidx] @ weights[3:5] + weights[5],
                   a_min=1.,
                   a_max=None)
    if verbose: print([alpha, beta])
    delta = datadict['deltas'][dataidx]
    n = datadict['ns'][dataidx]
    k = datadict['ks'][dataidx]
    if jacobian:
        data = np.hstack(
            [datadict['X'][dataidx], 1, datadict['X'][dataidx], 1.])
        dataDeriv = data[valueIdx]
        return (logpJacobianLinearAlpha if valueIdx < 3 else
                logpJacobianLinearBeta)(k, alpha, beta, delta, n, dataDeriv)
    return logp(k, alpha, beta, delta, n)


[(nd.Derivative(lambda x: f(x, widx, init, datadict))(init[widx]),
  f(init[widx], widx, init, datadict, True, verbose=True))
 for widx in range(6)]

[(nd.Derivative(lambda x: f(x, widx, init, datadict))(2),
  f(2, widx, init, datadict, True, verbose=True)) for widx in range(6)]
