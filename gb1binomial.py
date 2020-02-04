import theano
import theano.tensor as tt
from pymc3 import Discrete, floatX, intX
from pymc3.math import tround
from pymc3.distributions.bound import bound
from pymc3.distributions.discrete import binomln, betaln
from pymc3.distributions import draw_values, generate_samples


class Gb1Binomial(Discrete):
    "Adaptation of BetaBinomial"

    def __init__(self, alpha, beta, delta, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))
        self.delta = delta = tt.as_tensor_variable(floatX(delta))
        self.n = n = tt.as_tensor_variable(intX(n))
        # self.mode = tt.cast(tround(alpha / (alpha + beta)), 'int8')  # ??

    def _random(self, alpha, beta, n, delta, size=None):
        size = size or 1
        p = stats.beta.rvs(a=alpha, b=beta, size=size).flatten()
        # Sometimes scipy.beta returns nan. Ugh.
        while np.any(np.isnan(p)):
            i = np.isnan(p)
            p[i] = stats.beta.rvs(a=alpha, b=beta, size=np.sum(i))
        # Sigh...
        _n, _p, _size = np.atleast_1d(n).flatten(), p.flatten(), p.shape[0]
        _p = _p**delta

        quotient, remainder = divmod(_p.shape[0], _n.shape[0])
        if remainder != 0:
            raise TypeError(
                'n has a bad size! Was cast to {}, must evenly divide {}'.
                format(_n.shape[0], _p.shape[0]))
        if quotient != 1:
            _n = np.tile(_n, quotient)
        samples = np.reshape(stats.binom.rvs(n=_n, p=_p, size=_size), size)
        return samples

    def random(self, point=None, size=None):
        alpha, beta, n, delta = \
            draw_values([self.alpha, self.beta, self.n, self.delta], point=point, size=size)
        return generate_samples(self._random,
                                alpha=alpha,
                                beta=beta,
                                n=n,
                                delta=delta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        def mapper(n, k, alpha, beta, delta):
            i = tt.arange(n - k + 1.0)
            mags = binomln(n - k, i) + betaln(delta * (i + k) + alpha, beta)
            signs = (-tt.ones_like(i))**i  # (-1.0)**i
            return binomln(n, k) - betaln(alpha, beta) + logsumexp(mags, signs)

        return bound(
            theano.map(mapper,
                       sequences=[self.n, value],
                       non_sequences=[self.alpha, self.beta,
                                      self.delta])[0], value >= 0,
            value <= self.n, self.alpha > 0, self.beta > 0, self.delta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        alpha = dist.alpha
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Gb1Binomial}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(
            name, get_variable_name(alpha), get_variable_name(beta))


def logsumexp(x, signs, axis=None):
    "Adaptation of PyMC's logsumexp, but can take a list of signs like Scipy"
    x_max = tt.max(x, axis=axis, keepdims=True)
    result = tt.sum(signs * tt.exp(x - x_max), axis=axis, keepdims=True)
    return tt.switch(result >= 0,
                     tt.log(result) + x_max,
                     tt.log(-result) + x_max)
