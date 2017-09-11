""" Utilities for computing the probability mass function. """
import scipy.stats
import scipy.interpolate
import numpy
import tqdm
from fgivenx.parallel import openmp_apply, mpi_apply, rank
from fgivenx.io import CacheError, Cache


def PMF(samples, t=None):
    """ Compute the probability mass function.

        The set of samples defines a probability density P(t),
        which is computed using a kernel density estimator.

        From P(t) we define:

                  /
        M(p) =    | P(t) dt
                  /
              P(t) < p

        This is the cumulative distribution function expressed as a
        function of the probability

        We aim to compute M(t), which indicates the amount of
        probability contained outside the iso-probability contour
        passing through t.


         ^ P(t)                   ...
         |                     | .   .
         |                     |       .
        p|- - - - - - - - - - .+- - - - . - - - - - - - - - - -
         |                   .#|        #.
         |                  .##|        ##.
         |                  .##|        ##.
         |                 .###|        ###.     M(p)
         |                 .###|        ###.     is the
         |                 .###|        ###.     shaded area
         |                .####|        ####.
         |                .####|        ####.
         |              ..#####|        #####..
         |          ....#######|        #######....
         |         .###########|        ###########.
         +---------------------+-------------------------------> t
                              t

         ^ M(p)                        ^ M(t)
         |                             |
        1|                +++         1|         +
         |               +             |        + +
         |       ++++++++              |       +   +
         |     ++                      |     ++     ++
         |   ++                        |   ++         ++
         |+++                          |+++             +++
         +---------------------> p     +---------------------> t
        0                   1

        Parameters
        ----------
        samples: array-like
            Array of samples from a probability density P(t).
        
        t: array-like
            Array to evaluate the PDF at
    """
    # Compute the kernel density estimator from the samples
    samples = samples[~numpy.isnan(samples)]
    kernel = scipy.stats.gaussian_kde(samples)

    # Sort the samples in t, and find their probabilities
    samples = kernel.resample(10000)[0]
    samples.sort()
    ps = kernel(samples)

    # Compute the cumulative distribution function M(t) by
    # sorting the ps, and finding the position in that sort
    # We then store this as a log
    logms = numpy.log(scipy.stats.rankdata(ps) / float(len(samples)))

    samples
    # create an interpolating function of log(M(t))
    logpmf = scipy.interpolate.interp1d(samples, logms,
                                        bounds_error=False,
                                        fill_value=-numpy.inf)
    if t is not None:
        return numpy.exp(logpmf(t))
    else:
        return logpmf


def compute_masses(fsamps, y, **kwargs):
    """ Compute the masses at each x for a range of y.
    """
    parallel = kwargs.pop('parallel', '')
    nprocs = kwargs.pop('nprocs', None)
    comm = kwargs.pop('comm', None)
    cache = kwargs.pop('cache', None)


    if cache is not None:
        cache = Cache(cache + '_masses')
        try:
            return cache.check(fsamps, y)  
        except CacheError as e:
            print(e.args[0])

    if parallel is '':
        masses = [PMF(s, y) for s in tqdm.tqdm(fsamps)]
    elif parallel is 'openmp':
        masses = openmp_apply(PMF, fsamps, postcurry=(y,), nprocs=nprocs)
    elif parallel is 'mpi':
        masses = mpi_apply(lambda s: PMF(s, y), fsamps, comm=comm)
    else:
        raise ValueError("keyword parallel=%s not recognised,"
                         "options are 'openmp' or 'mpi'" % parallel)

    masses = numpy.array(masses).transpose().copy()

    if cache is not None and rank(comm) is 0:
        cache.data = fsamps, y, masses

    return masses
