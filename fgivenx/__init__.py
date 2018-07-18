r"""This module provides utilities for computing the grid for contours of a
    function reconstruction plot.

    Required ingredients:
     * posterior probability distribution :math:`P(\theta)` described by samples 
     * independent variable :math:`x`
     * dependent variable :math:`y`
     * functional form :math:`y = f(x;\theta)` parameterised by :math:`\theta`

    Assuming that you have obtained samples of theta from an MCMC
    process, we aim to compute the density:

    .. math::

        P(y|x) &= \int P(y=f(x;\theta)|x,\theta) P(\theta) d\theta \\
                &= \int \delta(y-f(x;\theta)) P(\theta) d\theta

    which gives our degree of knowledge for each :math:`y` value given an
    :math:`x` value.

    In fact, for a more representative plot, we are not actually
    interested in the value of the probability density above, but in fact
    require the "iso-probablity posterior mass"

    .. math::

        \mathrm{pmf}(y|x) = \int_{P(y'|x) < P(y|x)} P(y'|x) dy'

    We thus need to compute this function on a rectangular grid of :math:`x`
    and :math:`y`.
"""
import numpy
import fgivenx.samples
import fgivenx.mass
import fgivenx.dkl
from fgivenx._utils import _check_args, _normalise_weights


def compute_samples(f, x, samples, logZ=None, **kwargs):
    r"""
    Apply the function(s) :math:`f(x;\theta)` to the arrays defined in x and
    samples.  Has options for weighting, trimming, cacheing and parallelising.

    Additionally, if a list of log-evidences are passed, along with list of
    functions, samples and optional weights it marginalises over the models
    according to the evidences.

    Parameters
    ----------
    f: function
        function :math:`f(x;\theta)` with dependent variable :math:`x`,
        parameterised by :math:`\theta`.

    x: 1D array-like
        x values to evaluate :math:`f(x;\theta)` at.

    samples: 2D array-like
        theta samples to evaluate :math:`f(x;\theta)` at. 
        shape = (nsamples, npars)

    weights: 1D array-like, optional
        sample weights, if desired. Should have length same as samples.shape[0]
        default `numpy.ones_like(samples)`

    ntrim: int, optional
        Approximate number of samples to trim down to, if desired. Useful if
        the posterior is dramatically oversampled
        default None

    cache: str, optional
        File root for saving previous calculations for re-use
        default None

    parallel, tqdm_args:
        see docstring for :func:`fgivenx.parallel.parallel_apply`

    Returns
    -------
    2D numpy.array
        Evaluate the function f at each x value and each theta.
        Equivalent to `[[f(x_i,theta) for theta in samples] for x_i in x]`

    """
    weights = kwargs.pop('weights', None)
    parallel = kwargs.pop('parallel', False)
    ntrim = kwargs.pop('ntrim', None)
    cache = kwargs.pop('cache', None)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    logZ, f, x, samples, weights = _check_args(logZ, f, x, samples, weights)

    logZ, weights = _normalise_weights(logZ, weights, ntrim)

    for i, (s, w) in enumerate(zip(samples, weights)):
        samples[i] = fgivenx.samples.equally_weight_samples(s, w)

    return fgivenx.samples.compute_samples(f, x, samples,
                                           parallel=parallel, cache=cache,
                                           tqdm_kwargs=tqdm_kwargs)


def compute_pmf(f, x, samples, logZ=None, **kwargs):
    r"""
    Compute the probability mass function given x at a range of y values
    for :math:`y = f(x|\theta)`

    :math:`P(y|x) = \int P(y=f(x;\theta)|x,\theta) P(\theta) d\theta`

    :math:`\mathrm{pmf}(y|x) = \int_{P(y'|x) < P(y|x)} P(y'|x) dy'`

    Additionally, if a list of log-evidences are passed, along with list of
    functions, samples and optional weights it marginalises over the models
    according to the evidences.

    Parameters
    ----------
    f, x, samples, weights, ntrim, cache
        see arguments for :func:`fgivenx.compute_samples`

    ny: int
        Resolution of y axis

    y: array-like
        Explicit descriptor of y values to evaluate.

    tqdm_kwargs, parallel:
        see docstring for :func:`fgivenx.parallel.parallel_apply`

    Returns
    -------
    1D numpy.array:
        y values pmf is computed at `shape=(len(y))` or `ny`
    2D numpy.array:
        pmf values at each x and y  `shape=(len(x),len(y))`

    """

    weights = kwargs.pop('weights', None)
    parallel = kwargs.pop('parallel', False)
    ntrim = kwargs.pop('ntrim', 100000)
    ny = kwargs.pop('ny', 100)
    y = kwargs.pop('y', None)
    cache = kwargs.pop('cache', None)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # y
    if y is not None:
        y = numpy.array(y, dtype='double')
        if len(x.shape) is not 1:
            raise ValueError("y should be a 1D array")

    fsamps = compute_samples(f, x, samples, logZ=logZ,
                             weights=weights, ntrim=ntrim,
                             parallel=parallel, cache=cache,
                             tqdm_kwargs=tqdm_kwargs)

    if y is None:
        ymin = fsamps[~numpy.isnan(fsamps)].min(axis=None)
        ymax = fsamps[~numpy.isnan(fsamps)].max(axis=None)
        y = numpy.linspace(ymin, ymax, ny)

    return y, fgivenx.mass.compute_pmf(fsamps, y, parallel=parallel,
                                       cache=cache, tqdm_kwargs=tqdm_kwargs)


def compute_dkl(f, x, samples, prior_samples, logZ=None, **kwargs):
    r"""
    Compute the Kullback-Leibler divergence at each value of x for the prior
    and posterior defined by prior_samples and samples.

    Let the posterior be:

    :math:`P(y|x) = \int P(y=f(x;\theta)|x,theta)P(\theta) d\theta`

    and the prior be:

    :math:`Q(y|x) = \int P(y=f(x;\theta)|x,theta)Q(\theta) d\theta`

    then the Kullback-Leibler divergence at each x is defined by

    :math:`D_\mathrm{KL}(x) = \int P(y|x) \log\left[\frac{Q(y|x)}{P(y|x)}\right] dy`

    Parameters
    ----------
    f, x, samples, weights, ntrim, cache, parallel
        see arguments for :func:`fgivenx.compute_samples`

    Returns
    -------
    1D numpy array:
        dkl values at each value of x.
    """

    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', None)
    ntrim = kwargs.pop('ntrim', None)
    weights = kwargs.pop('weights', None)
    prior_weights = kwargs.pop('prior_weights', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if logZ is None:
        logZ = [0]
        f = [f]
        samples = [samples]
        prior_samples = [prior_samples]
        weights = [weights]
        prior_weights = [prior_weights]
        cache = [cache]

    DKLs = []

    for fi, c, s, w, ps, pw in zip(f, cache, samples, weights,
                                   prior_samples, prior_weights):

        fsamps = compute_samples(fi, x, s, weights=w, ntrim=ntrim,
                                 parallel=parallel, cache=c)

        fsamps_prior = compute_samples(fi, x, ps, weights=pw, ntrim=ntrim,
                                       parallel=parallel, cache=c+'_prior')

        dkls = fgivenx.dkl.compute_dkl(fsamps, fsamps_prior,
                                       parallel=parallel, cache=c)
        DKLs.append(dkls)

    logZ = numpy.array(logZ)
    DKLs = numpy.array(DKLs)

    Zs = numpy.exp(logZ-logZ.max())
    Zs /= Zs.sum()
    return numpy.sum(Zs * DKLs.transpose(), axis=1)
