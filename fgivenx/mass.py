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

        From :math:`P(y)` we define:

        :math:`\mathrm{pmf}(p) = \int_{P(y)<p} P(y) dy`

        This is the cumulative distribution function expressed as a
        function of the probability

        We aim to compute :math:`M(y)`, which indicates the amount of
        probability contained outside the iso-probability contour
        passing through :math:`y`::


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
            0

        Parameters
        ----------
        samples: array-like
            Array of samples from a probability density P(y).

        y: array-like (optional)
            Array to evaluate the PDF at

        Returns
        -------
        1D numpy.array:
            PMF evaluated at each y value

    """
    # Remove any nans from the samples
    samples = numpy.array(samples)
    samples = samples[~numpy.isnan(samples)]
    try:
        # Compute the kernel density estimate
        kernel = scipy.stats.gaussian_kde(samples)

        # Add two more samples definitely outside the range and sort them
        mn = min(samples) - 10*numpy.sqrt(kernel.covariance[0, 0])
        mx = max(samples) + 10*numpy.sqrt(kernel.covariance[0, 0])
        y_ = numpy.linspace(mn, mx, len(y)*10)

        # Compute the probabilities at each of the extended samples
        ps_ = kernel(y_)

        # Compute the masses
        ms = []
        for yi in y:
            # compute the probability at this y value
            p = kernel(yi)
            if p <= max(ps_)*1e-5:
                m = 0.
            else:
                # Find out which samples have greater probability than P(y)
                bools = ps_ > p

                # Compute indices where to start and stop the integration
                stops = numpy.where(numpy.logical_and(~bools[:-1], bools[1:]))
                starts = numpy.where(numpy.logical_and(bools[:-1], ~bools[1:]))

                # Compute locations
                starts = [scipy.optimize.brentq(lambda u: kernel(u)-p,
                                                y_[i], y_[i+1])
                          for i in starts[0]]
                starts = [-numpy.inf] + starts
                stops = [scipy.optimize.brentq(lambda u: kernel(u)-p,
                                               y_[i], y_[i+1])
                         for i in stops[0]]
                stops = stops + [numpy.inf]

                # Sum up the masses
                m = sum(kernel.integrate_box_1d(a, b)
                        for a, b in zip(starts, stops))
            ms.append(m)
        return numpy.array(ms)

    except numpy.linalg.LinAlgError:
        return numpy.zeros_like(y)

def PDF_hist(samples, y):
    """ Compute the probability density function (PDF) using a histogram.

        The set of samples defines a probability density P(y),
        which is computed using a histogram.

        Parameters
        ----------
        samples: array-like
            Array of samples from a probability density P(y).

        y: array-like (optional)
            Bin edges in between which to evaluate the PDF at.

        Returns
        -------
        1D numpy.array:
            PDF evaluated at each y bin `shape=(len(y)-1)`

    """
    # Compute a histogram of the samples, with bin edges specified by y
    edges = y
    hist, bin_edges = numpy.histogram(samples, bins=edges, density=True)
    assert numpy.all(edges == bin_edges), (
        "Bin edges different from specified:", edges, bin_edges)

    return hist

def PMF_hist(samples, y):
    """ Compute the probability mass function (PMF) using a histogram.

        The set of samples defines a probability density P(y),
        which is computed using a histogram. The PMF is computed
        for each bin specified by the bin edges y

        Parameters
        ----------
        samples: array-like
            Array of samples from a probability density P(y).

        y: array-like (optional)
            Bin edges in between which to evaluate the PDF at.

        Returns
        -------
        1D numpy.array:
            PMF evaluated at each y bin `shape=(len(y)-1)`

    """
    # Compute a histogram of the samples, with bin edges specified by y
    hist = PDF_hist(samples, y)

    # Integrate the PDF where PDF()<p for every p in [PDF(yi) for yi in y]
    bin_widths = numpy.diff(y)
    ms = []
    for i in range(len(y)-1):
        yi = y[i]
        p = hist[i]
        bools = hist < p
        m = numpy.sum((hist*bin_widths)[bools])
        ms.append(m)
    return numpy.array(ms)

def compute_pmf(fsamps, y, **kwargs):
    """ Compute the pmf defined by fsamps at each x for each y.

    Parameters
    ----------
    fsamps: 2D array-like
        array of function samples, as returned by
        :func:`fgivenx.compute_samples`

    y: 1D array-like
        y values to evaluate the PMF at

    histogram: bool, optional:
        Whether to estimate the probability mass function via a histogram
        instead of a kernel density estimate. Default: `False`

    pdf_histogram: bool, optional:
        Whether to actually compute the PDF instead of PMF. Default: `False`

    parallel, tqdm_kwargs: optional
        see docstring for :func:`fgivenx.parallel.parallel_apply`.

    Returns
    -------
    2D numpy.array
        probability mass function at each x for each y
        `shape=(len(fsamps),len(y)` except if histogram
        then `shape=(len(fsamps),len(y)-1)`
    """
    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', '')
    histogram = kwargs.pop('histogram', False)
    pdf_histogram = kwargs.pop('pdf_histogram', False)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache:
        cache = Cache(cache + '_masses')
        try:
            return cache.check(fsamps, y)
        except CacheException as e:
            print(e)

    # From a set of y-samples we compute the PDF(y). The we evaluate the PDF
    # at every given y to obtain p = [PDF(yi) for yi in y]. Then we compute
    # the PMF for each of probability-levels p to obtain integrals of the PDF
    # over within the iso-probability contour defined by each p.
    # Now do the whole thing for every x-slice using parallel_apply() which
    # takes a slice of fsamps (corresponding to a given x) and applies the
    # method above.
    # The histogram method differs slightly as we interpret y as bin edges
    # and evaluate everything within the bins.
    if not histogram:
        masses = parallel_apply(PMF, fsamps, postcurry=(y,), parallel=parallel,
                                tqdm_kwargs=tqdm_kwargs)
    elif not pdf_histogram:
        masses = parallel_apply(PMF_hist, fsamps, postcurry=(y,),
                                parallel=parallel, tqdm_kwargs=tqdm_kwargs)
    else:
        masses = parallel_apply(PDF_hist, fsamps, postcurry=(y,),
                                parallel=parallel, tqdm_kwargs=tqdm_kwargs)

    masses = numpy.array(masses).transpose().copy()

    if cache:
        cache.save(fsamps, y, masses)

    return masses
