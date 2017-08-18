""" fgivenx module

    Methods
    -------
    - samples_from_getdist_chains
    - compute_contours

    Example usage
    -------------
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
    file_root = 'chains/test'
    params = ['m','c']
    samples, weights = fgivenx.samples.samples_from_getdist_chains(file_root,
                                                                   params)
    x, y, z = fgivenx.compute_contours(f, x, samples, weights=weights)



"""
import numpy
import tqdm
from fgivenx.mass import PMF
from fgivenx.samples import compute_samples


def compute_contours(f, x, samples, **kwargs):
    """ Compute the contours ready for matplotlib plotting.

    Parameters
    ----------
    f : function
        f(x|theta)

    x : array-like
        Descriptor of x values to evaluate.

    samples: numpy.array
        2D Array of theta samples. samples.shape=(# of samples, len(theta),)

    Keywords
    --------
    """

    if not len(samples.shape) is 2:
        raise ValueError("samples should be a 2D numpy array")

    weights = kwargs.pop('weights', None)
    ntrim = kwargs.pop('ntrim', 0)
    ny = kwargs.pop('ny', 100)

    if weights is not None:
        samples = samples.trim_samples(samples, weights, ntrim)

    x = numpy.array(x)

    fsamps = compute_samples(f, x, samples)

    y = numpy.linspace(fsamps.min(), fsamps.max(), ny)

    z = compute_masses(fsamps, y)

    return x, y, z
