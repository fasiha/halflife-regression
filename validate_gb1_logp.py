from scipy.stats import beta, binom
from scipy.special import betaln
import numpy as np


def binomln(n, k):
    # https://stackoverflow.com/a/21775412/500207
    return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def genGb1Binom(Ndata, ALPHA, BETA, DELTA, N):
    n = np.ones(Ndata, dtype=int) * N
    p = beta.rvs(ALPHA, BETA, size=Ndata)**DELTA
    return binom.rvs(n, p)


ALPHA = 3.3
BETA = 4.4
DELTA = 3.33
N = 5

Ndata = 1_000_000
obs = genGb1Binom(Ndata, ALPHA, BETA, DELTA, N)


def logp(k, alpha, beta, delta, n):
    i = np.arange(n - k + 1.0)
    mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
    signs = (-1.0)**i
    return binomln(n, k) - betaln(alpha, beta) + logsumexp(mags, signs)


def logsumexp(x, signs):
    "Adaptation of PyMC's logsumexp, but can take a list of signs like Scipy"
    x_max = np.max(x)
    result = np.sum(signs * np.exp(x - x_max))
    return np.log(result if result >= 0 else -result) + x_max


def makeEmpiricalPmf(v):
    m = np.max(v)
    d = np.zeros(1 + m)
    for x in v:
        d[x] += 1
    return d / len(v)


pmf = np.array([np.exp(logp(n, ALPHA, BETA, DELTA, N)) for n in range(N + 1)])
assert np.allclose(np.sum(pmf), 1)
pmfhat = np.array(makeEmpiricalPmf(obs))
assert np.all(np.abs(pmfhat - pmf) / np.abs(pmf) < .1)