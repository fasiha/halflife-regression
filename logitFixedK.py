from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp, psi, expit


def binomln(n, k):
  # https://stackoverflow.com/a/21775412/500207
  return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def optimized(k, alpha, beta, delta, n, mu, kappa, logitMu, logitKappa, featureVecMu,
              featureVecKappa):
  i = np.arange(n - k + 1.0)
  ds = delta * (i + k)
  tosum = np.exp(binomln(n, k) + binomln(n - k, i) + betaln(ds + alpha, beta) - betaln(alpha, beta))
  tosum *= (-1.0)**i

  prob = np.sum(tosum)

  kappaPrime = kappa / (np.exp(logitKappa) + 1)
  muPrime = mu / (np.exp(logitMu) + 1)

  psiMu = psi(ds + alpha) - psi(alpha)
  baseJacMu = np.sum(psiMu * tosum) * muPrime / kappa

  psiKappa = -mu * psiMu + psi(ds + 1 / kappa) - psi(1 / kappa)
  baseJacKappa = np.sum(psiKappa * tosum) * kappaPrime / kappa**2
  if not np.isfinite(baseJacKappa):
    import pdb
    pdb.set_trace()
  return (prob, np.hstack([baseJacMu * featureVecMu, baseJacKappa * featureVecKappa]))


def optimizedObjective(weights, df):
  wm1, wm2, wm0, wk0 = weights
  logitMus = df.feature1 * wm1 + df.feature2 * wm2 + wm0
  mus = expit(logitMus)
  # if logitMu is too big, mu -> 1, beta -> 0
  logitKappas = np.ones(len(df)) * wk0
  kappas = expit(logitKappas)
  alphas = mus / kappas
  betas = (1 - mus) / kappas
  deltas = df.delta_.values
  ns = df.n.values
  ks = df.k.values

  totalY = 0
  totalJac = 0
  for (i,
       args) in enumerate(zip(ks, alphas, betas, deltas, ns, mus, kappas, logitMus, logitKappas)):
    featureVec = np.array([df.feature1.iloc[i], df.feature2.iloc[i], 1.0])
    y, jac = optimized(*args, featureVec, featureVec[-1:])
    totalY += y
    totalJac += jac
  return (-totalY, -totalJac)


def logp(k, alpha, beta, delta, n):
  i = np.arange(n - k + 1.0)
  mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
  signs = (-1.0)**i
  # if delta is 0.000005, logsumexp could be 0 (log0 = 1) or negative, causing underflow.
  # solutions: sqrt(delta), or clip here
  return binomln(n, k) - betaln(alpha, beta) + logsumexp(mags, b=signs, return_sign=False)


def pclip(p):
  # bound min/max model predictions (helps with loss optimization)
  return np.clip(p, a_min=0.0001, a_max=.99999)


def evaluate(weights, df):
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

  priorProbability = np.exp(betaln(alphas + deltas, betas) - betaln(alphas, betas))
  observedProbability = pclip(ks / ns)

  mae = np.mean(np.abs(priorProbability - observedProbability))

  # posteriors = np.exp(np.vectorize(logp)(ks, alphas, betas, deltas, ns))
  # meanPosterior = np.mean(posteriors)
  # quantiles = [0.1, 0.5, 0.9]
  # quantileValues = []  # np.quantile(posteriors, quantiles)

  return dict(meanAbsoluteError=mae)


def count2feature(x, log=True):
  return np.log(1 + x) + 1 if log else np.sqrt(1 + x)


fulldata = pd.read_csv("features.csv")  # features.csv for full
fulldata['feature1'] = count2feature(fulldata.history_correct)
fulldata['feature2'] = count2feature(fulldata.history_seen - fulldata.history_correct)
fulldata['delta_'] = np.sqrt(fulldata.delta / (3600 * 24))
fulldata['n'] = fulldata.session_seen
fulldata['k'] = fulldata.session_correct
fulldata['sqrtright'] = np.sqrt(1 + fulldata.history_correct)
fulldata['sqrtwrong'] = np.sqrt(1 + (fulldata.history_seen - fulldata.history_correct))


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
      if verbose and (xslice.start % verboseIteration) == 0:
        # prob = objective(weights, df)
        print("# Iteration {}/{:_}, weights={}, |grad|^2={:.1e}, Δ={:.1e}".format(
            t, xslice.start, weights, np.sum(grad**2), np.sqrt(np.sum(adjusted_grad**2))))
      xslice = slice(xslice.start + minibatchsize, xslice.stop + minibatchsize, xslice.step)

  return weights


def balance(df):
  ind0 = df[df.k < df.n].index
  ind1 = df[df.k == df.n].index
  N = min(len(ind0), len(ind1))
  idx = ind0[:N].append(ind1[:N]).sort_values()
  return df.loc[idx]


Ndata = round(len(fulldata) * 0.9)
# Ndata = 1000
data = fulldata[:Ndata]
balanced = balance(data)
test = fulldata[Ndata:]

init = np.zeros(4)
# w = adaGrad(init,data,minibatchsize=1000,max_it=3,stepsize=1., verboseIteration=10000)
# w = adaGrad(init,balanced,minibatchsize=1000,max_it=3,stepsize=1., verboseIteration=10000)
nice = np.array([0.19210431, -0.67477556, 2.35533034, 4.37681914])

nice2 = np.array([1.8806828, 1.02441168, 4.30784716, 5.721494])

# oh crap:
# evaluate( nice2,test[test.k<test.n])
# evaluate( nice2,test[test.k==test.n])