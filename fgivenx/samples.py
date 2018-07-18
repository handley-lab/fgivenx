import numpy
from fgivenx.parallel import parallel_apply
from fgivenx.io import CacheException, Cache
from fgivenx._utils import _equally_weight_samples



def compute_samples(f, x, samples, **kwargs):
    """ Apply f(x,theta) to x array and theta in samples.

    Parameters
    ----------
    See arguments of :func:`fgivenx.compute_contours`

    parallel, tqdm_kwargs: optional
        see docstring for :func:`fgivenx.parallel.parallel_apply`


    Returns
    -------
    2D numpy.array:
        samples at each x. `shape=(len(x),len(samples),)`
    """

    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', None)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache is not None:
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

    if cache is not None:
        cache.save(x, samples, fsamples)

    return fsamples


def samples_from_getdist_chains(params, file_root=None, chains_file=None,
                                paramnames_file=None, latex=False):
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

    chains_file: str, optional
        Full filename for getdist chains file.

    paramnames_file: str, optional
        Full filename for getdist paramnames file.

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

    # Get the full data
    if file_root is not None:
        if chains_file is not None:
            chains_file = file_root + '.txt'
        if paramnames_file is not None:
            paramnames_file = file_root + '.paramnames'

    if paramnames_file is None:
        raise ValueError("You must define paramnames_file,"
                         "either by file_root, or paramnames_file")
    if chains_file is None:
        raise ValueError("You must define chains_file,"
                         "either by file_root, or chains_file")

    data = numpy.loadtxt(chains_file)
    if len(data) is 0:
        return numpy.array([[]]), numpy.array([])
    if len(data.shape) is 1:
        data = data.reshape((1,) + data.shape)
    weights = data[:, 0]

    # Get the paramnames
    paramnames = [line.split()[0].replace('*', '')
                  for line in open(paramnames_file, 'r')]

    # Get the relevant samples
    indices = [2+paramnames.index(p) for p in params]
    samples = data[:, indices]

    if latex:
        latex = [' '.join(line.split()[1:])
                 for line in open(paramnames_file, 'r')]
        latex = [latex[i-2] for i in indices]
        return samples, weights, latex
    else:
        return samples, weights
