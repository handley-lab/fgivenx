""" fgivenx module.

    This module provides utilities for computing the grid for contours of a
    function reconstruction plot.

    Assume one has
     * posterior probability distribution P(theta) described by samples
     * independent variable x
     * dependent variable y
     * functional form y = f(x;theta) parameterised by theta

    Assuming that you have obtained samples of theta from an MCMC
    process, we aim to compute the density:

                  /
    P( y | x ) =  | P( y = f(x;theta) | x, theta ) dtheta ,  (1)
                  /

                  /
               =  | dirac_delta( y - f(x;theta) ) P(theta) dtheta ,  (2)
                  /

    which gives our degree of knowledge for each y value given an x value.

    In fact, for a more representative plot, we are not actually
    interested in the value of the probability density (1), but in fact
    require the "iso-probablity posterior mass:"

                        /
    m( y | x ) =        | P(y'|x) dy'
                        /
                P(y'|x) < P(y|x)

    We thus need to compute this function on a rectangular grid of x and y's.
"""
import numpy
import fgivenx.samples
from fgivenx.mass import compute_masses
from fgivenx.dkl import compute_dkl

def check_args(logZ, f, x, samples, weights):
    # convert to arrays
    if logZ is None:
        logZ = [0]
        f = [f]
        samples = [samples]
        weights = [weights]

    # logZ
    logZ = numpy.array(logZ, dtype='double')
    if len(logZ.shape) is not 1:
        raise ValueError("logZ should be a 1D array")

    # x
    x = numpy.array(x, dtype='double')
    if len(x.shape) is not 1:
        raise ValueError("x should be a 1D array")

    # f
    if len(logZ) != len(f):
            raise ValueError("len(logZ) = %i != len(f)= %i"
                             % (len(logZ), len(f)))
    for func in f:
        if not callable(func):
            raise ValueError("first argument f must be function"
                             "(or list of functions) of two variables")

    # samples
    if len(logZ) != len(samples):
            raise ValueError("len(logZ) = %i != len(samples)= %i"
                             % (len(logZ), len(samples)))
    samples = [numpy.array(s, dtype='double') for s in samples]
    for s in samples:
        if len(s.shape) is not 2:
            raise ValueError("each set of samples should be a 2D array")

    # weights
    if len(logZ) != len(weights):
            raise ValueError("len(logZ) = %i != len(weights)= %i"
                             % (len(logZ), len(weights)))
    weights = [numpy.array(w, dtype='double') if w is not None
               else numpy.ones(len(s), dtype='double')
               for w, s in zip(weights, samples)]

    for w, s in zip(weights, samples):
        if len(w.shape) is not 1:
            raise ValueError("each set of weights should be a 1D array")
        if len(w) != len(s):
            raise ValueError("len(w) = %i != len(s) = %i" % (len(s), len(w)))

    return logZ, f, x, samples, weights


def normalise_weights(logZ, weights, ntrim):
    Zs = numpy.exp(logZ-logZ.max())
    weights = [w/w.sum()*Z for w, Z in zip(weights, Zs)]
    wmax = max([w.max() for w in weights])
    weights = [w/wmax for w in weights]
    ntot = sum([w.sum() for w in weights])
    if ntrim is not None and ntrim < ntot:
        weights = [w*ntrim/ntot for w in weights]
    return logZ, weights


def compute_samples(f, x, samples, logZ=None, **kwargs):
    """
    Apply the function f(x;theta) 

    Parameters
    ----------
    x : array-like
        x values to evaluate f at.
    samples
    Keywords
    --------
    parallel:
        see docstring for fgivenx.parallel.parallel_apply.

    Returns
    -------
    """
    weights = kwargs.pop('weights', None)
    parallel = kwargs.pop('parallel', False)
    ntrim = kwargs.pop('ntrim', None)
    cache = kwargs.pop('cache', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    logZ, f, x, samples, weights  = check_args(logZ, f, x, samples, weights)

    logZ, weights = normalise_weights(logZ, weights, ntrim)

    for i, (s, w) in enumerate(zip(samples, weights)):
        samples[i] = fgivenx.samples.trim_samples(s, w)

    fsamps = fgivenx.samples.compute_samples(f, x, samples, parallel=parallel,
                                             cache=cache)

    return x, fsamps


def compute_contours(f, x, samples, logZ=None, **kwargs):
    """ Compute the contours ready for matplotlib plotting.

    Parameters
    ----------
    f : function or list of functions
        f(x|theta)
        if logZ is None: function
        if logZ is array-like: array-like of functions

    x : array-like
        Descriptor of x values to evaluate pmf at.

    samples: array-like
        if logZ is None: 2D array-likes
        if logZ is array-like: array-like of 2D arrays-likes


    logZ: array-like
        evidences to weight functions by


    Keywords
    --------
    weights: array-like
        Sample weights if samples are not equally weighted.
        len(weights) must equal len(samples)

    parallel:
        see docstring for fgivenx.parallel.parallel_apply.

    ntrim: int
        Number of samples to trim to (useful if your posterior is oversampled).

    ny: int
        Resolution of y axis

    y: array-like
        Explicit descriptor of y values to evaluate.

    cache: str
        Location to store cache files.

    Returns
    -------
    """

    weights = kwargs.pop('weights', None)
    parallel = kwargs.pop('parallel', False)
    ntrim = kwargs.pop('ntrim', 100000)
    ny = kwargs.pop('ny', 100)
    y = kwargs.pop('y', None)
    cache = kwargs.pop('cache', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    # y
    if y is not None:
        y = numpy.array(y, dtype='double')
        if len(x.shape) is not 1:
            raise ValueError("y should be a 1D array")

    x, fsamps = compute_samples(f, x, samples, logZ=logZ,
                                weights=weights, ntrim=ntrim,
                                parallel=parallel, cache=cache)

    if y is None:
        ymin = fsamps[~numpy.isnan(fsamps)].min(axis=None)
        ymax = fsamps[~numpy.isnan(fsamps)].max(axis=None)
        y = numpy.linspace(ymin, ymax, ny)

    z = compute_masses(fsamps, y, parallel=parallel, cache=cache)

    return x, y, z


def compute_kullback_liebler(f, x, samples, prior_samples, logZ=None, **kwargs):
    """
    Parameters
    ----------
    x : array-like
        x values to evaluate dkl at.
    Keywords
    --------
    parallel:
        see docstring for fgivenx.parallel.parallel_apply.

    Returns
    -------
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

        _, fsamps = compute_samples(fi, x, s, weights=w, ntrim=ntrim,
                                    parallel=parallel, cache=c)

        _, fsamps_prior = compute_samples(fi, x, ps, weights=pw, ntrim=ntrim,
                                          parallel=parallel, cache=c+'_prior')

        dkls = compute_dkl(x, fsamps, fsamps_prior, parallel=parallel, cache=c)
        DKLs.append(dkls)

    logZ = numpy.array(logZ)
    DKLs = numpy.array(DKLs)

    Zs = numpy.exp(logZ-logZ.max())
    Zs /= Zs.sum()
    return x, numpy.sum(Zs * DKLs.transpose(), axis=1)
