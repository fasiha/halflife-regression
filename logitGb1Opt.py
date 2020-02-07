import scipy.optimize as opt
from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp, psi, logit, expit
import numdifftools as nd
from functools import lru_cache


@lru_cache(maxsize=None)
def binomlnCached(n, k):
    # https://stackoverflow.com/a/21775412/500207
    k = np.array(k)
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


# Caching binomln (by converting array arguments to tuples) seems to be 1.08x.
def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def mylogsumexp(a, b):
    a_max = np.max(a)
    s = np.sum(b * np.exp(a - a_max))
    s = s if s >= 0 else -s
    return log(s) + a_max


def mysumexp(a, b):
    a_max = np.max(a)
    s = np.sum(b * np.exp(a - a_max))
    return s * np.exp(a_max)


def logp(k, alpha, beta, delta, n):
    i = np.arange(n - k + 1.0)
    mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
    signs = (-1.0)**i
    # if delta is 0.000005, logsumexp could be 0 (log0 = 1) or negative, causing underflow.
    # solutions: sqrt(delta), or clip here
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


def optimized(k, alpha, beta, delta, n, mu, kappa, logitMu, logitKappa,
              featureVec):
    i = np.arange(n - k + 1.0)
    ds = delta * (i + k)
    mags = binomln(n - k, i) + betaln(ds + alpha, beta) - betaln(alpha, beta)
    signs = (-1.0)**i
    denominator = np.exp(logsumexp(mags, b=signs))

    kappaPrime = kappa / (np.exp(logitKappa) + 1)
    muPrime = mu / (np.exp(logitMu) + 1)

    denominatorln = logsumexp(mags, b=signs)
    logp = denominatorln + binomln(n, k)

    psiMu = psi(ds + alpha) - psi(alpha)
    tmp = signs * np.exp(mags)

    numeratorMu = np.sum(psiMu * tmp) * muPrime / kappa
    baseJacMu = numeratorMu / denominator

    psiKappa = -mu * psiMu + psi(ds + 1 / kappa) - psi(1 / kappa)
    numeratorKappa = np.sum(psiKappa * tmp) * kappaPrime / kappa**2
    baseJacKappa = numeratorKappa / denominator

    return (logp,
            np.hstack([baseJacMu * featureVec, baseJacKappa * featureVec]))


def optimizedObjective(weights, df):
    wm1, wm2, wm0, wk1, wk2, wk0 = weights
    logitMus = df.feature1 * wm1 + df.feature2 * wm2 + wm0
    mus = expit(logitMus)
    logitKappas = df.feature1 * wk1 + df.feature2 * wk2 + wk0
    kappas = expit(logitKappas)
    alphas = mus / kappas
    betas = (1 - mus) / kappas
    deltas = df.delta_.values
    ns = df.n.values
    ks = df.k.values

    totalY = 0
    totalJac = 0
    for (i, args) in enumerate(
            zip(ks, alphas, betas, deltas, ns, mus, kappas, logitMus,
                logitKappas)):
        featureVec = np.array([df.feature1.iloc[i], df.feature2.iloc[i], 1.0])
        y, jac = optimized(*args, featureVec)
        totalY += y
        totalJac += jac
    return (-totalY, -totalJac)


def objective(weights, df, jacobian=False):
    wm1, wm2, wm0, wk1, wk2, wk0 = weights
    logitMus = df.feature1 * wm1 + df.feature2 * wm2 + wm0
    mus = expit(logitMus)
    logitKappas = df.feature1 * wk1 + df.feature2 * wk2 + wk0
    kappas = expit(logitKappas)
    alphas = mus / kappas
    betas = (1 - mus) / kappas
    deltas = df.delta_.values
    ns = df.n.values
    ks = df.k.values

    prob = sum(map(logp, ks, alphas, betas, deltas, ns))
    if jacobian:
        jac = np.array([
            sum(
                map(logpJacobianMu, ks, alphas, betas, deltas, ns, mus, kappas,
                    logitMus, logitKappas, df.feature1)),
            sum(
                map(logpJacobianMu, ks, alphas, betas, deltas, ns, mus, kappas,
                    logitMus, logitKappas, df.feature2)),
            sum(
                map(logpJacobianMu, ks, alphas, betas, deltas, ns, mus, kappas,
                    logitMus, logitKappas, np.ones(len(df)))),
            sum(
                map(logpJacobianKappa, ks, alphas, betas, deltas, ns, mus,
                    kappas, logitMus, logitKappas, df.feature1)),
            sum(
                map(logpJacobianKappa, ks, alphas, betas, deltas, ns, mus,
                    kappas, logitMus, logitKappas, df.feature2)),
            sum(
                map(logpJacobianKappa, ks, alphas, betas, deltas, ns, mus,
                    kappas, logitMus, logitKappas, np.ones(len(df)))),
        ])
        return (-prob, -jac)
    return -prob


def count2feature(x, log=True):
    return np.log(1 + x) + 1 if log else np.sqrt(1 + x)


million = pd.read_csv("pymcmill.csv")  # features.csv for full
million['feature1'] = count2feature(million.history_correct)
million['feature2'] = count2feature(million.history_seen -
                                    million.history_correct)
million['delta_'] = np.sqrt(million.delta / (3600 * 24))
million['n'] = million.session_seen
million['k'] = million.session_correct
Ndata = 1_000
data = million[:Ndata]


def verifyJacobian():
    def helper(x, w, i, data):
        w = w.copy()
        w[i] = x
        return objective(w, data)

    testVec = [1., 0., 1., 0., 1., 1.]
    print(objective(np.array(testVec), data))
    print(objective(np.array(testVec), data, jacobian=True))
    print(optimizedObjective(np.array(testVec), data))
    print([
        nd.Derivative(lambda x: helper(x, testVec, i, data))(xx)
        for (i, xx) in enumerate(testVec)
    ])


def evaluate(weights, df):
    wm1, wm2, wm0, wk1, wk2, wk0 = weights
    logitMus = df.feature1 * wm1 + df.feature2 * wm2 + wm0
    mus = expit(logitMus)
    logitKappas = df.feature1 * wk1 + df.feature2 * wk2 + wk0
    kappas = expit(logitKappas)
    alphas = mus / kappas
    betas = (1 - mus) / kappas
    deltas = df.delta_.values
    ns = df.n.values
    ks = df.k.values

    priorProbability = np.exp(
        betaln(alphas + deltas, betas) - betaln(alphas, betas))
    observedProbability = ks / ns

    mae = np.mean(np.abs(priorProbability - observedProbability))

    meanPosterior = np.mean(
        np.exp(np.vectorize(logp)(ks, alphas, betas, deltas, ns)))
    return dict(meanAbsoluteError=mae, meanPosterior=meanPosterior)


test = million[:Ndata]
evaluate(
    np.array([
        0.18302904, -0.68738957, 2.40760191, 0.93528212, 70.6042957,
        148.33726586
    ]), test)


def optim():
    init = np.array([1., 0., 1., 0., 1., 1.])
    init = np.array([
        0.31057201, -0.71416394, 2.27943638, 0.7567439, 151.24682103,
        277.99205638
    ])  # weights with 1000 points, regular delta
    init = np.array([
        0.21755801, -0.90975275, 2.8925863, 0.65805051, 75.58410503,
        175.98797187
    ])  # weights with 10_000 points, reguular delta
    # init = np.array([1., 0., 1., 0., 1., 1.])
    init = np.array([
        0.36361948, -0.66745148, 1.82226708, 0.78219012, 90.76710688,
        149.30608934
    ])  # 1_000 with sqrt days

    init = np.array([
        0.18302904, -0.68738957, 2.40760191, 0.93528212, 70.6042957,
        148.33726586
    ])  #  100_000 with sqrt days, took 3100 seconds
    import time
    start_time = time.time()
    sol = opt.minimize(lambda x: objective(x, data),
                       init,
                       method='Nelder-Mead',
                       options=dict(disp=True))
    print('auto-jacobian took {} seconds'.format(time.time() - start_time))
    print(sol)


np.seterr(all='raise')
# optim()
