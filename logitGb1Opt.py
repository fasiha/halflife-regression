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


def pclip(p):
    # bound min/max model predictions (helps with loss optimization)
    return np.clip(p, a_min=0.0001, a_max=.99999)


def hclip(h):
    # bound min/max half-life
    MIN_HALF_LIFE = 15.0 / (24 * 60)  # 15 minutes
    MAX_HALF_LIFE = 274.  # 9 months
    return np.clip(h, a_min=MIN_HALF_LIFE, a_max=MAX_HALF_LIFE)


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
fulldata['obsp'] = pclip(fulldata.session_correct / fulldata.session_seen)
fulldata['t'] = fulldata.delta / (60 * 60 * 24)  # convert time delta to days
fulldata['h'] = hclip(-fulldata.t / (np.log2(fulldata.obsp)))

Ndata = 1_000
data = fulldata[:Ndata]
test = fulldata[Ndata:]


def evaluateHLR(weights, df):
    dp = weights[0] * df.sqrtright + weights[1] * df.sqrtwrong + weights[2]
    h = hclip(2**dp)
    p = pclip(2.**(-df['t'] / h))

    mae = np.mean(np.abs(p - df.obsp))

    from scipy.stats import binom
    posteriors = binom.pmf(df.k, df.n, p)
    quantiles = [0.1, 0.5, 0.9]
    quantileValues = np.quantile(posteriors, quantiles)
    return dict(meanAbsoluteError=mae,
                meanPosterior=np.mean(posteriors),
                quantilePosterior=list(zip(quantiles, quantileValues)))


# evaluateHLR(np.array([-0.0125, -0.2245, 7.5365]), test)


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

    posteriors = np.exp(np.vectorize(logp)(ks, alphas, betas, deltas, ns))
    meanPosterior = np.mean(posteriors)
    quantiles = [0.1, 0.5, 0.9]
    quantileValues = np.quantile(posteriors, quantiles)

    return dict(meanAbsoluteError=mae,
                meanPosterior=meanPosterior,
                quantilePosterior=list(zip(quantiles, quantileValues)))


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

    init = np.zeros(6)
    solcg = opt.minimize(lambda w: optimizedObjective(w, data),
                         init,
                         options=dict(disp=True, gtol=1e-3),
                         tol=1e-3,
                         method='CG',
                         jac=True)
    evaluate(solcg.x, test)
    """Optimization terminated successfully.
            Current function value: 535.513650
            Iterations: 192
            Function evaluations: 437
            Gradient evaluations: 437
    """

    solBFGS = opt.minimize(lambda w: optimizedObjective(w, data),
                           init,
                           options=dict(disp=True, gtol=1e-3),
                           tol=1e-3,
                           method='BFGS',
                           jac=True)
    evaluate(solBFGS.x, test)
    """
    Optimization terminated successfully.
            Current function value: 535.513632
            Iterations: 55
            Function evaluations: 67
            Gradient evaluations: 67
    """

    iter = 0

    def callback(x):
        global iter
        iter += 1
        print("at iter {}".format(iter))

    solncg = opt.minimize(lambda w: optimizedObjective(w, data),
                          init,
                          options=dict(disp=True, xtol=1e-3),
                          tol=1e-3,
                          method='Newton-CG',
                          jac=True,
                          callback=callback)
    evaluate(solncg.x, test)
    """Optimization terminated successfully.
            Current function value: 535.545931
            Iterations: 12
            Function evaluations: 13
            Gradient evaluations: 72
            Hessian evaluations: 0"""


np.seterr(all='raise')

# optim()


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

init = np.zeros(6)
nice = np.array(
    [0.18648945, -0.70397773, 2.43953953, 0.75399696, 0.45243174, 0.30796359])
# w = adaGrad(init, data, minibatchsize=1000, max_it=3, stepsize=1., verboseIteration=1000)
# w = adaGrad(init, data, minibatchsize=1000, max_it=20_000, stepsize=1.)
# w = adaGrad(nice, data, minibatchsize=1000)
# w = adaGrad(w, data, minibatchsize=1000)

# evaluate(np.array([ 0.1833436 , -0.69785971,  2.38029033,  1.51137886 , 1.6426737 ,  1.57512674]), test)
# evaluate(np.array([ 0.21536606 ,-0.63705572 , 2.37135598 , 1.06562983 , 1.09257182 , 1.05931399 ]), test)
# evaluate(np.array([ 0.18325048 ,-0.67239645 , 2.39111254,  1.3849035  , 1.62478507  ,1.38028196 ]), test)

"20k iterations"
# evaluate(np.array([0.18491408,-0.69439215,  2.3902709 ,  1.17459025,  1.45161056 , 0.9900386]), test)
# evaluate(np.array([0.17480081 ,-0.66690311 , 2.38019202 , 1.15393516 , 1.46868852  ,0.91180955]), test)
