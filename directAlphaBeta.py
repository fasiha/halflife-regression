from scipy.special import betaln
import numpy as np
import pandas as pd
from scipy.special import logsumexp, psi, expit


def binomln(n, k):
  # https://stackoverflow.com/a/21775412/500207
  return -betaln(1 + n - k, 1 + k) - np.log(n + 1)


def optimized(k, alpha, beta, delta, n, alphaWeightsDotFeatureVec, betaWeightsDotFeatureVec,
              featureVec, expFormula):
  i = np.arange(n - k + 1.0)
  ds = delta * (i + k)
  tosum = np.exp(binomln(n, k) + binomln(n - k, i) + betaln(ds + alpha, beta) - betaln(alpha, beta))
  tosum *= (-1.0)**i

  prob = np.sum(tosum)

  psisBeta = psi(alpha + beta) - psi(alpha + beta + ds)
  psisAlpha = psisBeta + psi(alpha + ds) - psi(alpha)

  if expFormula:
    # `alpha = exp(weights @ featureVec)` model
    alphaPrime = alpha * featureVec
    betaPrime = beta * featureVec
  else:
    # `alpha = (weights @ featureVec)**2` model
    alphaPrime = alphaWeightsDotFeatureVec * 2 * featureVec
    betaPrime = betaWeightsDotFeatureVec * 2 * featureVec

  jacAlpha = np.sum(psisAlpha * tosum) * alphaPrime
  jacBeta = np.sum(psisBeta * tosum) * betaPrime
  jac = np.hstack([jacAlpha, jacBeta])
  if not np.all(np.isfinite(jac)):
    import pdb
    pdb.set_trace()
  return (prob, jac)


def evaluate(weights, df, expFormula):
  X = np.c_[df.feature1, df.feature2, np.ones(len(df))]
  alphaWeightsDotFeatureVec = X @ weights[:3]
  betaWeightsDotFeatureVec = X @ weights[3:]
  if expFormula:
    alphas = np.exp(alphaWeightsDotFeatureVec)
    betas = np.exp(betaWeightsDotFeatureVec)
  else:
    alphas = alphaWeightsDotFeatureVec**2
    betas = betaWeightsDotFeatureVec**2
  deltas = df.delta_.values
  ns = df.n.values
  ks = df.k.values

  priorProbability = np.exp(betaln(alphas + deltas, betas) - betaln(alphas, betas))
  observedProbability = pclip(ks / ns)

  mae = np.mean(np.abs(priorProbability - observedProbability))

  return dict(meanAbsoluteError=mae)


def optimizedObjective(weights, df, expFormula):
  X = np.c_[df.feature1, df.feature2, np.ones(len(df))]
  alphaWeightsDotFeatureVec = X @ weights[:3]
  betaWeightsDotFeatureVec = X @ weights[3:]
  if expFormula:
    alphas = np.exp(alphaWeightsDotFeatureVec)
    betas = np.exp(betaWeightsDotFeatureVec)
  else:
    alphas = alphaWeightsDotFeatureVec**2
    betas = betaWeightsDotFeatureVec**2
  deltas = df.delta_.values
  ns = df.n.values
  ks = df.k.values

  totalY = 0
  totalJac = 0
  for i in range(len(df)):
    n = ns[i]
    truek = ks[i]
    for k in range(n + 1):
      y, jac = optimized(
          k,
          alphas[i],
          betas[i],
          deltas[i],
          n,
          alphaWeightsDotFeatureVec[i],
          betaWeightsDotFeatureVec[i],
          X[i],
          expFormula=expFormula)
      if k == truek:
        continue
        totalY += y
        totalJac += jac
      else:
        totalY -= y
        totalJac -= jac
  return (-totalY, -totalJac)


def pclip(p):
  # bound min/max model predictions (helps with loss optimization)
  return np.clip(p, a_min=0.0001, a_max=.99999)


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
            objectiveFunction,
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
      val, grad = objectiveFunction(weights, sd)
      gti += grad**2
      adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
      weights -= stepsize * adjusted_grad
      if verbose and (xslice.start % verboseIteration) == 0:
        # prob = objective(weights, df)
        print("# Iteration {}/{:_}, weights={}, |grad|^2={:.1e}, Δ={:.1e}".format(
            t, xslice.start, weights, np.sum(grad**2), np.sqrt(np.sum(adjusted_grad**2))))
      xslice = slice(xslice.start + minibatchsize, xslice.stop + minibatchsize, xslice.step)

  return weights


Ndata = round(len(fulldata) * 0.9)
# Ndata = 1000
data = fulldata[:Ndata]
test = fulldata[Ndata:]

init = np.zeros(6)
# w = adaGrad(init,data,lambda w,df: optimizedObjective(w,df,True), minibatchsize=1000,max_it=3,stepsize=1., verboseIteration=1000)

wncg = np.array([1.51931747, 0.77973884, 0.58103441, -1.68240805, -0.86097934, -0.64768064])
wncgPM = np.array([1.9169825, 0.99362775, 0.77641418, -2.09854942, -1.08437096, -0.85206609])
# print(evaluate(wncgPM, test, True))
# w=wncg; cor=30; wro=2; x = np.sqrt(1+np.array([cor, wro, 0]));alpha=np.exp(x@w[:3]);beta=np.exp(x@w[3:]);alpha,beta


def verifyJacobian(expFormula=True):
  init = np.random.randn(6)
  init = np.zeros(6)
  small = fulldata[:1000]
  print(optimizedObjective(init, small, expFormula))
  import numdifftools as nd
  for widx, weight in enumerate(init):

    def foo(w):
      wvec = init.copy()
      wvec[widx] = w
      return optimizedObjective(wvec, small, expFormula)[0]

    print(nd.Derivative(foo)(weight))


iter = 0


def callback(x):
  global iter
  iter += 1
  print("at iter {}, x={}".format(iter, x))


import scipy.optimize as opt
solncg = opt.minimize(
    lambda w: optimizedObjective(w, data[:10000], True),
    init,
    options=dict(disp=True, xtol=1e-3),
    tol=1e-3,
    method='Newton-CG',
    jac=True,
    callback=callback)
