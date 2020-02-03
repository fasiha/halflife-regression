# %matplotlib inline
import numpy as np
import theano.tensor as tt
import pymc3 as pm

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import beta, binom

print('Running on PyMC3 v{}'.format(pm.__version__))

##

Ndata = 200

feature = np.random.rand(Ndata)
trials = np.random.randint(1, 20, (Ndata, ))

alphaSlope = 10.0
alphaIntercept = 5.0

betaSlope = -3.0
betaIntercept = 8.0

obs = [
    binom(n,
          beta(al, be).rvs(1)[0]).rvs(1)[0]
    for al, be, n in zip(alphaSlope * feature +
                         alphaIntercept, betaSlope * feature +
                         betaIntercept, trials)
]

##

import gb1binomial

with pm.Model() as model:
    # weights: slope, intercept
    walpha = pm.Uniform('walpha', lower=-20, upper=20, shape=2)
    wbeta = pm.Uniform('wbeta', lower=-20, upper=20, shape=2)

    # parameters of Beta
    a = (walpha[0] * feature + walpha[1]).clip(1, 5000)
    b = (wbeta[0] * feature + wbeta[1]).clip(1, 5000)

    # prior and likelihood
    x = gb1binomial.Gb1Binomial('x',
                                observed=obs,
                                n=trials,
                                alpha=a,
                                beta=b,
                                delta=1.0)

    trace = pm.sample(1000, tune=1000, cores=2)
# pm.traceplot(trace)
df = pm.trace_to_dataframe(trace)
print(df.describe())
