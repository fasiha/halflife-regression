import numpy as np
import theano.tensor as tt
import pymc3 as pm
from scipy.stats import beta, binom


def genGb1Binom(Ndata):
    a = 3.3
    b = 4.4
    d = .333
    n = np.random.randint(1, 20, size=Ndata)
    p = beta.rvs(a, b, size=Ndata)**d
    return (binom.rvs(n, p), n)


Ndata = 1000
obs, trials = genGb1Binom(Ndata)

import gb1binomial

with pm.Model() as model:
    a = pm.Uniform('a', lower=1, upper=10)
    b = pm.Uniform('b', lower=1, upper=10)
    d = pm.Uniform('d', lower=.01, upper=10)

    x = gb1binomial.Gb1Binomial('x',
                                observed=obs,
                                n=trials,
                                alpha=a,
                                beta=b,
                                delta=d)

    trace = pm.sample(2000, tune=1000, cores=2)
df = pm.trace_to_dataframe(trace)
print(df.describe())
