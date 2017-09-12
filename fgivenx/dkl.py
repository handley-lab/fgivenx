import tqdm
import numpy
from scipy.stats import gaussian_kde
from fgivenx.io import CacheError, Cache
from fgivenx.parallel import parallel_apply

def dkl(arrays):
    """
    Compute the Kullback-Liebler divergence from samples from prior and posterior.
    Parameters
    ----------
    Keywords
    --------
    Returns
    -------
    """
    samples, prior_samples = arrays
    samples = samples[~numpy.isnan(samples)]
    prior_samples = prior_samples[~numpy.isnan(prior_samples)]
    return (
            gaussian_kde(samples).logpdf(samples) 
            - gaussian_kde(prior_samples).logpdf(samples)
            ).mean()


def compute_dkl(x, fsamps, prior_fsamps, **kwargs):
    """
    Parameters
    ----------
    Keywords
    --------
    Returns
    -------
    """

    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache is not None:
        cache = Cache(cache + '_dkl')
        try:
            return cache.check(x, fsamps, prior_fsamps)  
        except CacheError as e:
            print(e.msg())

    zip_fsamps = list(zip(fsamps, prior_fsamps))
    dkls = parallel_apply(dkl, zip_fsamps, parallel=parallel)
    dkls = numpy.array(dkls)

    if cache is not None:
        cache.save(x, fsamps, prior_fsamps, dkls)

    return dkls

