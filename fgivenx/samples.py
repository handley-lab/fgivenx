import numpy
from fgivenx.parallel import parallel_apply
from fgivenx.io import CacheException, Cache


def compute_samples(f, x, samples, **kwargs):
    r""" Apply f(x,theta) to x array and theta in samples.

    Parameters
    ----------
    f: function
        list of functions :math:`f(x;\theta)`  with dependent variable
        :math:`x`, parameterised by :math:`\theta`.

    x: 1D array-like
        x values to evaluate :math:`f(x;\theta)` at.

    samples: 2D array-like
        list of theta samples to evaluate :math:`f(x;\theta)` at.
        `shape = (nfunc, nsamples, npars)`

    parallel, tqdm_kwargs: optional
        see docstring for :func:`fgivenx.parallel.parallel_apply`

    cache: str, optional
        File root for saving previous calculations for re-use
        default None

    Returns
    -------
    2D numpy.array:
        samples at each x. `shape=(len(x),len(samples),)`
    """

    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', '')
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache:
        cache = Cache(cache + '_fsamples')
        try:
            return cache.check(x, samples)
        except CacheException as e:
            print(e)

    fsamples = []
    for fi, s in zip(f, samples):
        if len(s) > 0:
            fsamps = parallel_apply(fi, s, precurry=(x,), parallel=parallel,
                                    tqdm_kwargs=tqdm_kwargs)
            fsamps = numpy.array(fsamps).transpose().copy()
            fsamples.append(fsamps)
    fsamples = numpy.concatenate(fsamples, axis=1)

    if cache:
        cache.save(x, samples, fsamples)

    return fsamples


def samples_from_getdist_chains(params, file_root, latex=False):
    """ Extract samples and weights from getdist chains.

    Parameters
    ----------
    params: list(str)
        Names of parameters to be supplied to second argument of f(x|theta).

    file_root: str, optional
        Root name for getdist chains files. This variable automatically
        defines:
        - chains_file = file_root.txt
        - paramnames_file = file_root.paramnames
        but can be overidden by chains_file or paramnames_file.

    latex: bool, optional
        Also return an array of latex strings for those paramnames.

    Returns
    -------
    samples: numpy.array
        2D Array of samples. `shape=(len(samples), len(params))`

    weights: numpy.array
        Array of weights. `shape = (len(params),)`

    latex: list(str), optional
        list of latex strings for each parameter
        (if latex is provided as an argument)
    """

    import getdist
    samples = getdist.loadMCSamples(file_root)
    weights = samples.weights

    indices = [samples.index[p] for p in params]
    samps = samples.samples[:, indices]
    if latex:
        latex = [samples.parLabel(p) for p in params]
        return samps, weights, latex
    else:
        return samps, weights
