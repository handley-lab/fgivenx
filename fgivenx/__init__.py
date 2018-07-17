r""" fgivenx module.

    This module provides utilities for computing the grid for contours of a
    function reconstruction plot.

    Required ingredients:
     * posterior probability distribution :math:`P(\theta)` described by samples 
     * independent variable :math:`x`
     * dependent variable :math:`y`
     * functional form :math:`y = f(x;\theta)` parameterised by :math:`\theta`

    Assuming that you have obtained samples of theta from an MCMC
    process, we aim to compute the density:

    .. math::

        P(y|x) &= \int P(y=f(x;\theta)|x,\theta)d\theta \\
                &= \int \delta(y-f(x;\theta) P(\theta) d\theta

    which gives our degree of knowledge for each :math:`y` value given an
    :math:`x` value.

    In fact, for a more representative plot, we are not actually
    interested in the value of the probability density above, but in fact
    require the "iso-probablity posterior mass:"

    .. math::

        \mathrm{pmf}(y|x) = \int_{P(y'|x) < P(y|x)} P(y'|x) dy'

    We thus need to compute this function on a rectangular grid of :math:`x`
    and :math:`y`.
"""
import numpy
import fgivenx.samples
import fgivenx.mass
import fgivenx.dkl


def compute_samples(f, x, samples, logZ=None, **kwargs):
    """
    Apply the function(s) f(x;theta) to the arrays defined in x and samples.
    Has options for weighting, trimming, cacheing and parallelising.

    Additionally, if a list of log-evidences are passed, along with list of
    functions, samples and optional weights it marginalises over the models
    according to the evidences.

    Parameters
    ----------
    f: function
        function f(x;theta) with dependent variable x, parameterised by theta.

    x: 1D array-like
        x values to evaluate f(x;theta) at.

    samples: 2D array-like
        theta samples to evaluate f(x;theta) at. shape = (nsamples, npars)

    Keywords
    --------
    weights: 1D array-like
        sample weights, if desired. Should have length same as samples.shape[0]

    ntrim: int
        Approximate number of samples to trim down to, if desired. Useful if
        the posterior is dramatically oversampled.

    cache: str
        File root for saving previous calculations for re-use.

    parallel:
        see docstring for fgivenx.parallel.parallel_apply.

    tqdm_kwargs: dict
        see docstring for fgivenx.parallel.parallel_apply.

    Returns
    -------
    2D numpy array
        Evaluate the function f at each x value and each theta.
        Equivalent to [[f(x_i,theta) for theta in samples] for x_i in x]

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
    """
    Compute the probablity mass function given x at a range of y values
    for y = f(x|theta)

                  /
    P( y | x ) =  | P( y = f(x;theta) | x, theta ) dtheta
                  /

                          /
    pmf( y | x ) =        | P(y'|x) dy'
                          /
                  P(y'|x) < P(y|x)

    Additionally, if a list of log-evidences are passed, along with list of
    functions, samples and optional weights it marginalises over the models
    according to the evidences.

    Parameters
    ----------
    f, x, samples, weights, ntrim, cache, parallel
        see arguments for fgivenx.compute_samples

    ny: int
        Resolution of y axis

    y: array-like
        Explicit descriptor of y values to evaluate.

    Keywords
    --------
    tqdm_kwargs: dict
        see docstring for fgivenx.parallel.parallel_apply.

    Returns
    -------
    1D numpy array:
        y values pmf is computed at
    2D numpy array:
        pmf values at each x and y

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
    """
    Compute the Kullback-Liebler divergence at each value of x for the prior
    and posterior defined by prior_samples and samples.

    Let the posterior be:

                  /
    P( y | x ) =  | P( y = f(x;theta) | x, theta ) dtheta
                  /

    and prior be:

                  /
    Q( y | x ) =  | Prior( y = f(x;theta) | x, theta ) dtheta
                  /

    then the Kullback-Liebler divergence at each x is defined by:

              /
    D_KL(x) = | P( y | x ) log( Q( y | x ) / P( y | x ) ) dy
              /

    Parameters
    ----------
    f, x, samples, weights, ntrim, cache, parallel
        see arguments for fgivenx.compute_samples

    Keywords
    --------
    parallel:
        see docstring for fgivenx.parallel.parallel_apply.

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


def _check_args(logZ, f, x, samples, weights):
    """ Check the arguments for compute_samples. """
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


def _normalise_weights(logZ, weights, ntrim):
    """ Correctly normalise the weights for trimming"""
    Zs = numpy.exp(logZ-logZ.max())
    weights = [w/w.sum()*Z for w, Z in zip(weights, Zs)]
    wmax = max([w.max() for w in weights])
    weights = [w/wmax for w in weights]
    ntot = sum([w.sum() for w in weights])
    if ntrim is not None and ntrim < ntot:
        weights = [w*ntrim/ntot for w in weights]
    return logZ, weights
