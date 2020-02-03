import theano.tensor as tt
from pymc3 import Discrete, floatX, intX
from pymc3.math import tround
from pymc3.distributions.bound import bound
from pymc3.distributions.discrete import binomln, betaln
from pymc3.distributions import draw_values, generate_samples


class Gb1Binomial(Discrete):
    def __init__(self, alpha, beta, delta, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))
        self.delta = delta = tt.as_tensor_variable(floatX(delta))
        self.n = n = tt.as_tensor_variable(intX(n))
        self.mode = tt.cast(tround(alpha / (alpha + beta)), 'int8')  # ??

    def _random(self, alpha, beta, n, size=None):
        size = size or 1
        p = stats.beta.rvs(a=alpha, b=beta, size=size).flatten()
        # Sometimes scipy.beta returns nan. Ugh.
        while np.any(np.isnan(p)):
            i = np.isnan(p)
            p[i] = stats.beta.rvs(a=alpha, b=beta, size=np.sum(i))
        # Sigh...
        _n, _p, _size = np.atleast_1d(n).flatten(), p.flatten(), p.shape[0]
        _p = p**self.delta

        quotient, remainder = divmod(_p.shape[0], _n.shape[0])
        if remainder != 0:
            raise TypeError(
                'n has a bad size! Was cast to {}, must evenly divide {}'.
                format(_n.shape[0], _p.shape[0]))
        if quotient != 1:
            _n = np.tile(_n, quotient)
        samples = np.reshape(stats.binom.rvs(n=_n, p=_p, size=_size), size)
        return samples

    # def random(self, point=None, size=None): # try calling parent?
    def random(self, point=None, size=None):
        """
        Draw random values from BetaBinomial distribution.
        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).
        Returns
        -------
        array
        """
        alpha, beta, n = \
            draw_values([self.alpha, self.beta, self.n], point=point, size=size)
        return generate_samples(self._random,
                                alpha=alpha,
                                beta=beta,
                                n=n,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of BetaBinomial distribution at specified value.
        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor
        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta
        delta = self.delta
        return bound(
            binomln(self.n, value) +
            betaln(value + alpha, self.n - value + beta) - betaln(alpha, beta),
            value >= 0, value <= self.n, alpha > 0, beta > 0, delta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        alpha = dist.alpha
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Gb1Binomial}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(
            name, get_variable_name(alpha), get_variable_name(beta))
