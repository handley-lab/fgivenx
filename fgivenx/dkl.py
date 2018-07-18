import numpy
from scipy.stats import gaussian_kde
from fgivenx.io import CacheException, Cache
from fgivenx.parallel import parallel_apply


def DKL(arrays):
    """
    Compute the Kullback-Leibler divergence from one distribution Q to another
    P, where Q and P are represented by a set of samples.

    Parameters
    ----------
    arrays: tuple(1D numpy.array,1D numpy.array)
        samples defining distributions P & Q respectively

    Returns
    -------
    float:
        Kullback Leibler divergence.
    """
    samples, prior_samples = arrays
    samples = samples[~numpy.isnan(samples)]
    prior_samples = prior_samples[~numpy.isnan(prior_samples)]
    return (
            gaussian_kde(samples).logpdf(samples)
            - gaussian_kde(prior_samples).logpdf(samples)
            ).mean()


def compute_dkl(fsamps, prior_fsamps, **kwargs):
    """
    Compute the Kullback Leibler divergence for function samples for posterior
    and prior pre-calculated at a range of x values.

    Parameters
    ----------
    fsamps: 2D numpy.array
        Posterior function samples, as computed by :func:`fgivenx.compute_samples`

    prior_fsamps: 2D numpy.array
        Prior function samples, as computed by :func:`fgivenx.compute_samples`

    parallel, tqdm_kwargs: optional
        see docstring for :func:`fgivenx.parallel.parallel_apply`.

    cache: str, optional
        File root for saving previous calculations for re-use.

    Returns
    -------
    1D numpy.array:
        Kullback-Leibler divergences at each value of x. `shape=(len(fsamps))`
    """

    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache is not None:
        cache = Cache(cache + '_dkl')
        try:
            return cache.check(fsamps, prior_fsamps)
        except CacheException as e:
            print(e)

    zip_fsamps = list(zip(fsamps, prior_fsamps))
    dkls = parallel_apply(DKL, zip_fsamps, parallel=parallel)
    dkls = numpy.array(dkls)

    if cache is not None:
        cache.save(fsamps, prior_fsamps, dkls)

    return dkls
