""" Utilities for computing the probability mass function. """
import scipy.stats
import scipy.interpolate
import numpy
from fgivenx.parallel import parallel_apply
from fgivenx.io import CacheException, Cache


def PMF(samples, t=None):
    """ Compute the probability mass function.

        The set of samples defines a probability density P(t),
        which is computed using a kernel density estimator.

        From P(t) we define:

                    /
        PMF(p) =    | P(t) dt
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

        t: array-like (optional)
            Array to evaluate the PDF at

        Returns
        -------
        if t == None:
            function for the log of the pmf
        else:
            PMF evaluated at each t value

    """
    # Compute the kernel density estimator from the samples
    samples = numpy.array(samples)
    samples = samples[~numpy.isnan(samples)]
    try:
        kernel = scipy.stats.gaussian_kde(samples)

        # Sort the samples in t, and find their probabilities
        samples = kernel.resample(10000)[0]
        samples.sort()
        ps = kernel(samples)

        # Compute the cumulative distribution function M(t) by
        # sorting the ps, and finding the position in that sort
        # We then store this as a log
        logms = numpy.log(scipy.stats.rankdata(ps) / float(len(samples)))

        # create an interpolating function of log(M(t))
        logpmf = scipy.interpolate.interp1d(samples, logms,
                                            bounds_error=False,
                                            fill_value=-numpy.inf)
    except numpy.linalg.LinAlgError:
        # If the samples all have approximately the same value (for example
        # this can occur if the function you are plotting converges) then
        # scipy.stats.gaussian_kde(samples) may throw a LinAlgError when the
        # standard deviation of the samples goes to zero (to within numerical
        # accuracy).
        # In this case return an interpolating function that is 1 exactly on
        # the samples and zero elsewhere.
        # NB pmf=1 implies logpmf=0.
        if numpy.std(samples, ddof=1) == 0:
            logpmf = scipy.interpolate.interp1d(samples,
                                                numpy.zeros(samples.shape),
                                                bounds_error=False,
                                                fill_value=-numpy.inf)
        else:
            raise numpy.linalg.LinAlgError("numpy.linalg.LinAlgError not " +
                                           "handeled as samples std != 0")
    if t is not None:
        return numpy.exp(logpmf(numpy.array(t)))
    else:
        return logpmf


def compute_pmf(fsamps, y, **kwargs):
    """ Compute the pmf defined by fsamps at each x for each y.

    Parameters
    ----------
    fsamps: 2D array-like
        array of function samples, as returned by fgivenx.compute_samples

    y: 1D array-like
        y values to evaluate the PMF at

    Keywords
    --------
    parallel: bool
        see docstring for fgivenx.parallel.parallel_apply.

    tqdm_kwargs: dict
        see docstring for fgivenx.parallel.parallel_apply.

    Returns
    -------
    """
    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', None)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache is not None:
        cache = Cache(cache + '_masses')
        try:
            return cache.check(fsamps, y)
        except CacheException as e:
            print(e)

    masses = parallel_apply(PMF, fsamps, postcurry=(y,), parallel=parallel,
                            tqdm_kwargs=tqdm_kwargs)
    masses = numpy.array(masses).transpose().copy()

    if cache is not None:
        cache.save(fsamps, y, masses)

    return masses
