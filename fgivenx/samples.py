import numpy
from fgivenx.parallel import parallel_apply
from fgivenx.io import CacheError, Cache


def equally_weight_samples(samples, weights):
    """ Convert samples to be equally weighted.

    Samples are trimmed by discarding samples in accordance with a probability
    determined by the corresponding weight.

    This function has assumed you have normalised the weights properly.
    If in doubt, convert weights via:
    weights /= weights.max() .

    Parameters
    ----------
    samples: array-like
        Samples to trim.

    weights: array-like
        Weights to trim by.

    Returns
    -------
    1D numpy.array
    Equally weighted sample array.
    """
    if len(weights) != len(samples):
        raise ValueError("len(weights) = %i != len(samples) = %i" %
                         (len(weights), len(samples)))

    if numpy.logical_or(weights < 0, weights > 1).any():
        raise ValueError("weights must have probability between 0 and 1")

    weights = numpy.array(weights)
    samples = numpy.array(samples)

    state = numpy.random.get_state()

    numpy.random.seed(1)
    n = len(weights)
    choices = numpy.random.rand(n) < weights

    new_samples = samples[choices]

    numpy.random.set_state(state)

    return new_samples.copy()


def compute_samples(f, x, samples, **kwargs):
    """ Apply f(x,theta) to x array and theta in samples.

    Parameters
    ----------
    See arguments of fgivenx.compute_contours

    Keywords
    --------
    parallel:
        see docstring for fgivenx.parallel.parallel_apply.

    Returns
    -------
    An array of samples at each x. shape=(len(x),len(samples),)
    """

    parallel = kwargs.pop('parallel', False)
    cache = kwargs.pop('cache', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if cache is not None:
        cache = Cache(cache + '_fsamples')
        try:
            return cache.check(x, samples)
        except CacheError as e:
            print(e)

    fsamples = []
    for fi, s in zip(f, samples):
        if len(s) > 0:
            fsamps = parallel_apply(fi, s, precurry=x, parallel=parallel)
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

    Keywords
    --------
    file_root: str
        Root name for getdist chains files. This variable automatically
        defines:
        - chains_file = file_root.txt
        - paramnames_file = file_root.paramnames
        but can be overidden by chains_file or paramnames_file.

    chains_file: str
        Full filename for getdist chains file.

    paramnames_file: str
        Full filename for getdist paramnames file.

    latex: bool
        Also return an array of latex strings for those paramnames.

    Returns
    -------
    samples: numpy.array
        2D Array of samples. samples.shape=(# of samples, len(params),)

    weights: numpy.array
        Array of weights. samples.shape = (len(params),)
    if latex:
    latex: list(str)
        list of latex strigs for each parameter
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
