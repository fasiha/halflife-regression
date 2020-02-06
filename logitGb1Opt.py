import scipy.optimize as opt
from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp, psi, logit, expit


def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def logp(k, alpha, beta, delta, n):
    i = np.arange(n - k + 1.0)
    mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
    signs = (-1.0)**i
    return binomln(n, k) - betaln(alpha, beta) + logsumexp(
        mags, b=signs, return_sign=False)


def objective(weights, datadict):
    mus = expit(datadict['X'] @ weights[:2] + weights[2])
    ks = expit(datadict['X'] @ weights[3:5] + weights[5])
    alphas = mus / ks
    betas = (1 - mus) / ks
    deltas = datadict['deltas']
    ns = datadict['ns']
    ks = datadict['ks']
    return -sum(map(logp, ks, alphas, betas, deltas, ns))


def count2feature(x, log=True):
    return np.log(1 + x) + 1 if log else np.sqrt(1 + x)


million = pd.read_csv("pymcmill.csv")  # features.csv for full
million['scaledright'] = count2feature(million.history_correct)
million['scaledwrong'] = count2feature(million.history_seen -
                                       million.history_correct)
million['days'] = million.delta / (3600 * 24)
data = million[:1000]

datadict = dict(X=np.array([data.scaledright.values,
                            data.scaledwrong.values]).T,
                deltas=data.days.values,
                ns=data.session_seen.values,
                ks=data.session_correct.values)

init = np.array([7., -30, 48, 1, 2, 3])
init = np.array([1., 0., 1., 0., 1., 1.])
sol = opt.minimize(lambda x: objective(x, datadict),
                   init,
                   method='Nelder-Mead')
print(sol)
# np.seterr(all='raise')
# objective(init, datadict)
