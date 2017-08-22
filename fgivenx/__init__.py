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

    Example usage
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    import fgivenx
    import numpy
    import matplotlib.pyplot
    import fgivenx.plot


    # Define a simple straight line function, parameters theta=(m,c)
    def f(x, theta):
        m, c = theta
        return m * x + c

    # Create some sample gradient and intercepts
    nsamples = 1000
    ms = numpy.random.normal(loc=1,size=nsamples)
    cs = numpy.random.normal(loc=0,size=nsamples)
    samples = numpy.array([(m,c) for m,c in zip(ms,cs)])

    # Examine the function over a range of x's
    xmin, xmax = -2, 2
    nx = 100
    x = numpy.linspace(xmin, xmax, nx)

    # Compute the contours
    x, y, z = fgivenx.compute_contours(f, x, samples)

    # Plot
    fig, ax = matplotlib.pyplot.subplots()
    cbar = fgivenx.plot.plot(x, y, z, ax)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    Samples can also be computed from getdist chains using the helper function
    `fgivenx.samples.samples_from_getdist_chains`:


    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    params = ['m','c']
    file_root = 'chains/test'
    samples, weights = fgivenx.samples.samples_from_getdist_chains(params,
                                                                   file_root)
    x, y, z = fgivenx.compute_contours(f, x, samples, weights=weights)



"""
import numpy
from fgivenx.mass import compute_masses
from fgivenx.samples import compute_samples, trim_samples
from fgivenx.io import SampleCache, DKLCache
from fgivenx.dkl import compute_dkl


def compute_contours(f, x, samples, **kwargs):
    """ Compute the contours ready for matplotlib plotting.

    Parameters
    ----------
    f : function
        f(x|theta)

    x : array-like
        Descriptor of x values to evaluate.

    samples: array-like
        2D Array of theta samples. shape should be (# of samples, len(theta),)

    Keywords
    --------
    weights: array-like
        Sample weights if samples are not equally weighted.
        len(weights) must equal len(samples)

    parallel: str
        Type of parallelisation to use. Must be either 'openmp' or 'mpi'.

    ntrim: int
        Number of samples to trim to (useful if your posterior is oversampled)

    ny: int
        Resolution of y axis

    y: array-like
        Explicit descriptor of y values to evaluate.

    cache: str
        Location to store cache files.
    """

    weights = kwargs.pop('weights', None)
    parallel = kwargs.pop('parallel', '')
    ntrim = kwargs.pop('ntrim', 0)
    ny = kwargs.pop('ny', 100)
    y = kwargs.pop('y', None)
    nprocs = kwargs.pop('nprocs', None)
    comm = kwargs.pop('comm', None)
    cache = kwargs.pop('cache',None)
    prior = kwargs.pop('prior',False)

    # Argument checking
    # =================
    # f
    if not callable(f):
        raise ValueError("first argument f must be function of two variables")

    # samples
    samples = numpy.array(samples, dtype='double')
    if len(samples.shape) is not 2:
        raise ValueError("samples should be a 2D array")

    # x
    x = numpy.array(x, dtype='double')
    if len(x.shape) is not 1:
        raise ValueError("x should be a 1D array")

    # weights
    if weights is not None:
        weights = numpy.array(weights, dtype='double')
        if len(weights) != len(samples):
            raise ValueError("length of samples (%i) != length of weights (%i)"
                             % (len(samples), len(weights)))
    else:
        weights = numpy.ones(len(samples), dtype='double')

    # y
    if y is not None:
        y = numpy.array(y, dtype='double')
        if len(x.shape) is not 1:
            raise ValueError("y should be a 1D array")

    #cache 
    if cache is not None:
        cache = SampleCache(cache)
    # Computation
    # ===========

    samples = trim_samples(samples, weights, ntrim)

    fsamps = compute_samples(f, x, samples, parallel=parallel,
                             nprocs=nprocs, comm=comm, cache=cache)

    if y is None:
        y = numpy.linspace(fsamps.min(), fsamps.max(), ny)

    z = compute_masses(fsamps, y, parallel=parallel,
                       nprocs=nprocs, comm=comm, cache=cache, prior=prior)

    return x, y, z

def compute_kullback_liebler(f, x, samples, prior_samples, **kwargs):

    nprocs = kwargs.pop('nprocs', None)
    parallel = kwargs.pop('parallel', '')
    comm = kwargs.pop('comm', None)
    cache = kwargs.pop('cache',None)
    ntrim = kwargs.pop('ntrim', 0)
    weights = kwargs.pop('weights', None)
    prior_weights = kwargs.pop('prior_weights', None)

    cache = DKLCache(cache)
    samples = trim_samples(samples, weights, ntrim)
    prior_samples = trim_samples(prior_samples, prior_weights, ntrim)

    fsamps = compute_samples(f, x, samples, parallel=parallel,
                             nprocs=nprocs, comm=comm, cache=cache.posterior(),
                             ntrim=ntrim, weights=weights)

    fsamps_prior = compute_samples(f, x, prior_samples, parallel=parallel,
                                   nprocs=nprocs, comm=comm, cache=cache.prior(),
                                   ntrim=ntrim, weights=weights)

    dkls = compute_dkl(x, fsamps, fsamps_prior, parallel=parallel,
                       nprocs=nprocs, comm=comm, cache=cache) 

    return x, dkls
