""" Utilities for computing the probability mass function. """
import scipy.stats
import scipy.interpolate
import numpy
from fgivenx.parallel import parallel_apply
from fgivenx.io import CacheException, Cache


def PMF(samples, y):
    """ Compute the probability mass function.

        The set of samples defines a probability density P(y),
        which is computed using a kernel density estimator.

        From P(y) we define:

                    /
        PMF(p) =    | P(t) dt
                    /
                P(t) < p

        This is the cumulative distribution function expressed as a
        function of the probability

        We aim to compute M(y), which indicates the amount of
        probability contained outside the iso-probability contour
        passing through y.


         ^ P(y)                   ...
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
         +---------------------+-------------------------------> y
                              t

         ^ M(p)                        ^ M(y)
         |                             |
        1|                +++         1|         +
         |               +             |        + +
         |       ++++++++              |       +   +
         |     ++                      |     ++     ++
         |   ++                        |   ++         ++
         |+++                          |+++             +++
         +---------------------> p     +---------------------> y
        0                   1

        Parameters
        ----------
        samples: array-like
            Array of samples from a probability density P(y).

        y: array-like (optional)
            Array to evaluate the PDF at

        Returns
        -------
        PMF evaluated at each y value

    """
    # Remove any nans from the samples
    samples = numpy.array(samples)
    samples = samples[~numpy.isnan(samples)]
    try:
        # Compute the kernel density estimate
        kernel = scipy.stats.gaussian_kde(samples)

        # Add two more samples definitely outside the range and sort them
        mn = 1.5*min(samples) - 0.5*max(samples)
        mx = 1.5*max(samples) - 0.5*min(samples)
        samples_ = numpy.array([mn, mx] + list(samples))
        samples_.sort()

        # Compute the probabilities at each of the extended samples
        ps_ = kernel(samples_)

        # Compute the masses
        ms = []
        for yi in y:
            # Zero mass if it's outside the range
            if yi < mn or yi > mx:
                m = 0.
            else:

                # compute the probability at this y value
                p = kernel(yi)

                # Find out which samples have greater probability than P(y)
                bools = ps_>p

                # Compute indices where to start and stop the integration
                stops = numpy.where(numpy.logical_and(~bools[:-1], bools[1:]))[0]
                starts = numpy.where(numpy.logical_and(bools[:-1], ~bools[1:]))[0]

                # Compute locations
                starts =  [mn] + [scipy.optimize.brentq(lambda u: kernel(u)-p,samples_[i], samples_[i+1]) for i in starts]
                stops = [scipy.optimize.brentq(lambda u: kernel(u)-p,samples_[i], samples_[i+1]) for i in stops] + [mx]

                # 
                m = sum(kernel.integrate_box_1d(a, b) for a, b in zip(starts, stops))
            ms.append(m)
        return numpy.array(ms)

    except numpy.linalg.LinAlgError:
        return numpy.zeros_like(y)

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
