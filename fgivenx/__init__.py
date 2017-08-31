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
from fgivenx.io import Cache
from fgivenx.dkl import compute_dkl


def compute_contours(f, x, samples, **kwargs):
    """ Compute the contours ready for matplotlib plotting.

    Parameters
    ----------
    f : function or list of functions
        f(x|theta)

    x : array-like
        Descriptor of x values to evaluate.

    samples: array-like or list of array-like
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

    logZ: array-like
        evidences to weight functions by
    """

    weights = kwargs.pop('weights', None)
    parallel = kwargs.pop('parallel', '')
    ntrim = kwargs.pop('ntrim', -1)
    ny = kwargs.pop('ny', 100)
    y = kwargs.pop('y', None)
    nprocs = kwargs.pop('nprocs', None)
    comm = kwargs.pop('comm', None)
    cache = kwargs.pop('cache',None)
    prior = kwargs.pop('prior',False)
    logZs = kwargs.pop('logZs',False) 

    # Argument checking
    # =================
    # f
    if not logZs:
        logZs = [0]
        f = [f]
        samples = [samples]
        weights = [weights]

    if [i for i in f if not callable(i)]:
        raise ValueError("first argument f must be function (or list of functions) of two variables")

    # samples
    samples = [numpy.array(s, dtype='double') for s in samples]
    if [i for i in samples if len(i.shape) is not 2]:
        raise ValueError("samples should be a 2D array")

    # x
    if len(x.shape) is not 1:
        raise ValueError("x should be a 1D array")

    # weights
    weights = [numpy.array(i, dtype='double') if i is not None 
               else numpy.ones(len(s), dtype='double')  
               for i, s in zip(weights, samples)]

    for w, s in zip(weights,samples):
        if len(w) != len(s):
            raise ValueError("length of samples (%i) != length of weights (%i)"
                             % (len(s), len(w)))

    # y
    if y is not None:
        y = numpy.array(y, dtype='double')
        if len(x.shape) is not 1:
            raise ValueError("y should be a 1D array")

    logZs = numpy.array(logZs)

    # Computation
    # ===========
    Zs = numpy.exp(logZs-logZs.max())
    weights = [w/w.sum()*Z for w, Z in zip(weights,Zs)]
    wmax = max([w.max() for w in weights])
    weights = [w/wmax for w in weights]
    ntot = sum([w.sum() for w in weights])
    if ntrim < ntot:
        weights = [w*ntrim/ntot for w in weights]


    for i, (s, w) in enumerate(zip(samples, weights)):
        samples[i] = trim_samples(s, w)

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
    logZs = kwargs.pop('logZs',False) 

    if not logZs:
        logZs = [0]
        f = [f]
        samples = [samples]
        prior_samples = [prior_samples]
        weights = [weights]
        cache = [cache]

    DKLs = []

    for fi, c, s, p, w in zip(f, cache, samples, prior_samples, weights):
        fsamps = compute_samples([fi], x, [s], parallel=parallel,
                                 nprocs=nprocs, comm=comm, cache=c,
                                 ntrim=ntrim, weights=w)

        fsamps_prior = compute_samples([fi], x, [p], parallel=parallel,
                                       nprocs=nprocs, comm=comm, cache=c + '_prior',
                                       ntrim=ntrim, weights=w)

        dkls = compute_dkl(x, fsamps, fsamps_prior, parallel=parallel,
                           nprocs=nprocs, comm=comm, cache=c) 
        DKLs.append(dkls)

    logZs = numpy.array(logZs)
    DKLs = numpy.array(DKLs)

    Zs = numpy.exp(logZs-logZs.max())
    Zs /= Zs.sum()
    return x, numpy.sum(Zs * DKLs.transpose(), axis=1)
