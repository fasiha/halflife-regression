import scipy.optimize as opt
from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp, psi, logit, expit
import numdifftools as nd
from functools import lru_cache


def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def optimized(k, alpha, beta, delta, n, mu, kappa, logitMu, logitKappa,
              featureVecMu, featureVecKappa):
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
            np.hstack(
                [baseJacMu * featureVecMu, baseJacKappa * featureVecKappa]))


def optimizedObjective(weights, df):
    wm1, wm2, wm0, wk0 = weights
    logitMus = df.feature1 * wm1 + df.feature2 * wm2 + wm0
    mus = expit(logitMus)
    logitKappas = np.ones(len(df)) * wk0
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
        y, jac = optimized(*args, featureVec, featureVec[-1:])
        totalY += y
        totalJac += jac
    return (-totalY, -totalJac)


def count2feature(x, log=True):
    return np.log(1 + x) + 1 if log else np.sqrt(1 + x)


fulldata = pd.read_csv("features.csv")  # features.csv for full
fulldata['feature1'] = count2feature(fulldata.history_correct)
fulldata['feature2'] = count2feature(fulldata.history_seen -
                                     fulldata.history_correct)
fulldata['delta_'] = np.sqrt(fulldata.delta / (3600 * 24))
fulldata['n'] = fulldata.session_seen
fulldata['k'] = fulldata.session_correct
fulldata['sqrtright'] = np.sqrt(1 + fulldata.history_correct)
fulldata['sqrtwrong'] = np.sqrt(1 + (fulldata.history_seen -
                                     fulldata.history_correct))


# https://github.com/benbo/adagrad/blob/master/adagrad.py
def adaGrad(weights,
            df,
            stepsize=1e-2,
            fudge_factor=1e-6,
            max_it=2,
            minibatchsize=250,
            verbose=True,
            verboseIteration=1000):
    weights = weights.copy()
    gti = np.zeros_like(weights)

    for t in range(max_it):
        df = df.sample(frac=1.0)
        xslice = slice(0, minibatchsize)
        while xslice.start < len(df):
            sd = df.sample(minibatchsize).reset_index(
                drop=True)  # https://stackoverflow.com/a/34879805/500207
            val, grad = optimizedObjective(weights, sd)
            gti += grad**2
            adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
            weights -= stepsize * adjusted_grad
            if verbose:
                # prob = objective(weights, df)
                print(
                    "# Iteration {}/{:_}, weights={}, |grad|^2={:.1e}, Δ={:.1e}"
                    .format(t, xslice.start, weights, np.sum(grad**2),
                            np.sqrt(np.sum(adjusted_grad**2))))
            xslice = slice(xslice.start + minibatchsize,
                           xslice.stop + minibatchsize, xslice.step)

    return weights


Ndata = round(len(fulldata) * 0.9)
data = fulldata[:Ndata]
test = fulldata[Ndata:]

init = np.zeros(4)
w = adaGrad(init,
            data,
            minibatchsize=1000,
            max_it=3,
            stepsize=1.,
            verboseIteration=1000)
