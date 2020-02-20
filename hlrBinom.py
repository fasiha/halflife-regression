import numpy as np
import scipy.stats as stats
import scipy.special as special


def prob(k, n, t, x, w):
  h = 2**(x @ w)
  p = 2**(-t / h)
  return stats.binom.pmf(k, n, p)


def evaluate(w, df):
  x = np.c_[df.sqrtright, df.sqrtwrong, np.ones(len(df)), df.sqrttotal]
  h = 2**(x @ w)
  p = 2**(-df.t / h)
  pmf = stats.binom.pmf(df.k, df.n, p)

  mae = np.mean(np.abs(p - df.obsp))
  return dict(meanAbsoluteError=mae, meanPosterior=np.mean(pmf))


def binomln(n, k):
  # https://stackoverflow.com/a/21775412/500207
  return -special.betaln(1 + n - k, 1 + k) - np.log(n + 1)


def probJacobian(k, n, t, x, w, widx):
  h = 2**(x @ w)
  p = 2**(-t / h)
  pmf = stats.binom.pmf(k, n, p)
  return pmf * x[widx] * np.log(2)**2 * t / (1 - p) * (k - n * p) / h


log2 = np.log(2)
loglog2Times2 = 2 * np.log(log2)


def probJacobianAccurate(k, n, t, x, w, widx):
  logh = x @ w * log2
  h = np.exp(logh)
  logp = -t / h * log2
  log1MinusP = np.log(-np.expm1(logp))
  logpmf = binomln(n, k) + k * logp + (n - k) * log1MinusP
  logretbase = (logpmf + loglog2Times2 + np.log(t) - logh - log1MinusP + np.log(x[widx]))
  return np.exp(logretbase) * k - np.exp(logretbase + logp + np.log(n))


def testAccuracy():
  probJacobian(0, 1, 0.001597, np.array([16.822604, 12.041595, 1]),
               np.array([1.5386902, 1.60667677, 1.69713884]), 0)
  probJacobianAccurate(0, 1, 0.001597, np.array([16.822604, 12.041595, 1]),
                       np.array([1.5386902, 1.60667677, 1.69713884]), 0)
  sumProbJac(0, 1, 0.001597, np.array([16.822604, 12.041595, 1]),
             np.array([1.5386902, 1.60667677, 1.69713884]))


def sumProbJac(k, n, t, x, w):
  h = 2**(x @ w)
  p = 2**(-t / h)
  pmf = stats.binom.pmf(k, n, p)
  jacBase = pmf * np.log(2)**2 * t / (1 - p) * (k - n * p) / h
  return (-np.sum(pmf), -(np.atleast_1d(jacBase) @ x))


def mysumexp(a, b=1):
  a_max = np.max(a)
  s = np.sum(b * np.exp(a - a_max))
  return s * np.exp(a_max)


def sumProbJacDf(w, df, uselog=True):
  x = np.c_[df.sqrtright, df.sqrtwrong, np.ones(len(df)), df.sqrttotal]
  logh = (x @ w) * log2
  h = np.exp(logh)
  logp = (-df.t / h) * log2
  log1MinusP = np.log(-np.expm1(logp))
  logpmf = binomln(df.n, df.k) + df.k * logp + (df.n - df.k) * log1MinusP

  logretbase = logpmf + loglog2Times2 + np.log(df.t) - logh - log1MinusP
  jacBase = np.exp(logretbase) * df.k - np.exp(logretbase + logp + np.log(df.n))
  if not uselog:
    return (-mysumexp(logpmf), -(jacBase @ x))

  # D[log(f(x)), x] = f'(x) / f(x)
  logfx = logsumexp(logpmf)
  fx = np.exp(logfx)
  fprimex = jacBase @ x
  logfprimex = fprimex / fx
  return (-logfx, -(logfprimex))


def verify(data, uselog=True):
  init = np.zeros(4)
  small = data[:1000]
  print(sumProbJacDf(init, small, uselog))
  import numdifftools as nd
  for widx, weight in enumerate(init):

    def foo(w):
      wvec = init.copy()
      wvec[widx] = w
      return sumProbJacDf(wvec, small, uselog)[0]

    print(nd.Derivative(foo)(weight))


import pandas as pd
from scipy.special import logsumexp
fulldata = pd.read_csv("features.csv")
fulldata['n'] = fulldata.session_seen
fulldata['k'] = fulldata.session_correct
fulldata['t'] = fulldata.delta / (60 * 60 * 24)  # convert time delta to days
fulldata['sqrtright'] = np.sqrt(1 + fulldata.history_correct)
fulldata['sqrtwrong'] = np.sqrt(1 + (fulldata.history_seen - fulldata.history_correct))
fulldata['sqrttotal'] = np.sqrt(1 + fulldata.history_seen)
fulldata['right'] = fulldata.history_correct
fulldata['logright'] = np.log(1 + fulldata.right)
fulldata['wrong'] = fulldata.history_seen - fulldata.history_correct
fulldata['logwrong'] = np.log(1 + fulldata.wrong)
fulldata[
    'obsp'] = fulldata.session_correct / fulldata.session_seen  # clip these to compare to HLR paper?

Ndata = round(len(fulldata) * .9)
data = fulldata[:Ndata]
test = fulldata[Ndata:]


# https://github.com/benbo/adagrad/blob/master/adagrad.py
def adaGrad(weights,
            origDf,
            stepsize=1e-2,
            fudge_factor=1e-6,
            max_it=2,
            minibatchsize=250,
            verbose=True,
            verboseIteration=100,
            toBalance=False):
  weights = weights.copy()
  gti = np.zeros_like(weights)

  for t in range(max_it):
    df = origDf.sample(frac=1.0)
    if toBalance:
      df = balance(df)
    # x = np.c_[df.sqrtright, df.sqrtwrong, np.ones(len(df))]
    xslice = slice(0, minibatchsize)
    while xslice.start < len(df):
      # https://stackoverflow.com/a/34879805/500207
      sd = df[xslice]
      # sx = x[xslice, :]
      val, grad = sumProbJacDf(weights, sd)
      gti += grad**2
      adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
      weights -= stepsize * adjusted_grad
      if verbose and (xslice.start % verboseIteration == 0):
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


# np.seterr(all='raise')
init = np.zeros(3)
# w = adaGrad(init, data, stepsize=1., max_it=2, minibatchsize=10000,verboseIteration=100_000, toBalance=True)
balanced = balance(data)
# w = adaGrad(init, balanced, stepsize=1., max_it=50, minibatchsize=10000,verboseIteration=100_000)

wnice = np.array([3.88079004, 4.50591716, 5.18645383])
wbal = np.array([1.67063654, -5.60241933, 9.82442276])


def optim(data, rebalance=True, init=None):
  import scipy.optimize as opt
  if init is None:
    init = np.zeros(4)
  iter = 0

  def callback(x):
    nonlocal iter
    iter += 1
    print("at iter {}, x={}".format(iter, x.tolist()))

  df = balance(data) if rebalance else data
  print('data len', len(df))
  solncg = opt.minimize(
      lambda w: sumProbJacDf(w, df),
      init,
      options=dict(disp=True, xtol=1e-3),
      tol=1e-3,
      method='Newton-CG',
      jac=True,
      callback=callback)


# print(optim(data.sample(frac=1.0)))
# at iter 45, x=[ 2.0144001  -6.87423151 11.57950714] # this is maximizing p, not log(p)
# at iter 8, x=[2.079918567916871, -7.07191546094335, 11.746564528780242] # maximizing log(p), different sample
# at iter 7, x=[1.6238849597801754, -5.552162163881284, 9.82985120557989] # maximizing log(p) but no shuffling, i.e., log doesn't change optimization. I feel stupid :P.
# at iter 64, x=[7.84411111648866, -3.166204175724058, 9.728716263991107, -6.661347708048041] with rebalance, random shuffle
# at iter 64, x=[6.568406934249408, -3.8884813903126116, 10.030539798096072, -5.226503120116738]


def paramsToHalflife(cor, wro, w):
  return 2**(np.sqrt([cor + 1, wro + 1, 1]) @ w)


def paramsToHalflife4(cor, wro, w):
  return 2**(np.sqrt([cor + 1, wro + 1, 1, 1 + wro + cor]) @ w)


wpaper = np.array([-0.0125, -0.2245, 7.5365])
