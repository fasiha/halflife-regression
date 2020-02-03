# %matplotlib inline
import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import beta, binom

print('Running on PyMC3 v{}'.format(pm.__version__))

##

Ndata = 1000


def genBetaBinom(Ndata):
    feature = np.random.rand(Ndata)
    trials = np.random.randint(1, 20, (Ndata, ))

    alphaSlope = 10.0
    alphaIntercept = 5.0

    betaSlope = -3.0
    betaIntercept = 8.0

    return ([
        binom(n,
              beta(al, be).rvs(1)[0]).rvs(1)[0]
        for al, be, n in zip(alphaSlope * feature +
                             alphaIntercept, betaSlope * feature +
                             betaIntercept, trials)
    ], trials)


def genGb1Binom(Ndata):
    a = 3.3
    b = 4.4
    d = .333
    n = np.random.randint(1, 20, size=Ndata)
    p = beta.rvs(a, b, size=Ndata)**d
    return (binom.rvs(n, p), n)


obs, trials = genGb1Binom(Ndata)

##

import gb1binomial

with pm.Model() as model:
    # weights: slope, intercept
    a = pm.Uniform('a', lower=1, upper=10)
    b = pm.Uniform('b', lower=1, upper=10)
    d = pm.Uniform('d', lower=.01, upper=10)

    # prior and likelihood
    x = gb1binomial.Gb1Binomial('x',
                                observed=obs,
                                n=trials,
                                alpha=a,
                                beta=b,
                                delta=d)

    trace = pm.sample(2000, tune=1000, cores=2)
# pm.traceplot(trace)
df = pm.trace_to_dataframe(trace)
print(df.describe())
