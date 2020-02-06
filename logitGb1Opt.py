import scipy.optimize as opt
from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp, psi, logit, expit
import numdifftools as nd


def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def logp(k, alpha, beta, delta, n):
    i = np.arange(n - k + 1.0)
    mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
    signs = (-1.0)**i
    return binomln(n, k) - betaln(alpha, beta) + logsumexp(
        mags, b=signs, return_sign=False)


def logpJacobianMu(k, alpha, beta, delta, n, mu, kappa, logitMu, logitKappa,
                   feature):
    muPrime = mu / (np.exp(logitMu) + 1) * feature
    i = np.arange(n - k + 1.0)
    ds = delta * (i + k)
    mags = binomln(n - k, i) + betaln(ds + alpha, beta)
    signs = (-1.0)**i
    Bab = betaln(alpha, beta)
    denominator = np.exp(logsumexp(mags, b=signs) - Bab)
    numerator = np.sum((psi(ds + alpha) - psi(alpha)) * muPrime / kappa *
                       signs * np.exp(mags - Bab))
    return numerator / denominator


def logpJacobianKappa(k, alpha, beta, delta, n, mu, kappa, logitMu, logitKappa,
                      feature):
    kappaPrime = kappa / (np.exp(logitKappa) + 1) * feature
    i = np.arange(n - k + 1.0)
    ds = delta * (i + k)
    mags = binomln(n - k, i) + betaln(ds + alpha, beta)
    signs = (-1.0)**i
    Bab = betaln(alpha, beta)
    denominator = np.exp(logsumexp(mags, b=signs) - Bab)
    psis = mu * (psi(alpha) - psi(ds + alpha)) + psi(ds + 1 / kappa) - psi(
        1 / kappa)
    numerator = np.sum(psis * kappaPrime / kappa**2 * signs *
                       np.exp(mags - Bab))
    return numerator / denominator


def derivTest(w, muNotKappa=True, jacobian=False):
    feature = -0.3
    logitMu = 0.2 + (w * feature if muNotKappa else 0)
    mu = expit(logitMu)
    logitKappa = 1.2 + (w * feature if not muNotKappa else 0)
    kappa = expit(logitKappa)
    alpha = mu / kappa
    beta = (1 - mu) / kappa
    n = 4
    k = 4
    delta = 0.8
    if not jacobian: return logp(k, alpha, beta, delta, n)

    return (logpJacobianMu if muNotKappa else logpJacobianKappa)(
        k, alpha, beta, delta, n, mu, kappa, logitMu, logitKappa, feature)


pt = 2.1
[
    nd.Derivative(derivTest)(pt),
    derivTest(pt, jacobian=True),
    nd.Derivative(lambda x: derivTest(x, muNotKappa=False))(pt),
    derivTest(pt, muNotKappa=False, jacobian=True)
]


def objective(weights, datadict, jacobian=False):
    logitMus = datadict['X'] @ weights[:2] + weights[2]
    mus = expit(logitMus)
    logitKappas = datadict['X'] @ weights[3:5] + weights[5]
    ks = expit(logitKappas)
    alphas = mus / ks
    betas = (1 - mus) / ks
    deltas = datadict['deltas']
    ns = datadict['ns']
    ks = datadict['ks']

    prob = sum(map(logp, ks, alphas, betas, deltas, ns))
    if jacobian:
        jac = np.array([
            sum(
                map(logpJacobianMu, ks, alphas, betas, deltas, ns, mus, kappas,
                    logitMus, logitKappas, datadict['X'][:, 0])),
            sum(
                map(logpJacobianMu, ks, alphas, betas, deltas, ns, mus, kappas,
                    logitMus, logitKappas, datadict['X'][:, 1])),
            sum(
                map(logpJacobianMu, ks, alphas, betas, deltas, ns, mus, kappas,
                    logitMus, logitKappas, np.ones(len(data)))),
            sum(
                map(logpJacobianKappa, ks, alphas, betas, deltas, ns, mus,
                    kappas, logitMus, logitKappas, datadict['X'][:, 0])),
            sum(
                map(logpJacobianKappa, ks, alphas, betas, deltas, ns, mus,
                    kappas, logitMus, logitKappas, datadict['X'][:, 1])),
            sum(
                map(logpJacobianKappa, ks, alphas, betas, deltas, ns, mus,
                    kappas, logitMus, logitKappas, np.ones(len(data)))),
        ])
        return (-prob, -jac)
    return -prob


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


def optim():
    init = np.array([1., 0., 1., 0., 1., 1.])
    import time
    start_time = time.time()
    sol = opt.minimize(lambda x: objective(x, datadict),
                       init,
                       method='Nelder-Mead',
                       options=dict(disp=True))
    print('auto-jacobian took {} seconds'.format(time.time() - start_time))
    print(sol)


optim()
