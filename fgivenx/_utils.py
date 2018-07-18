import numpy


def _check_args(logZ, f, x, samples, weights):
    """ Sanity-check the arguments for compute_samples.

    Parameters
    ----------
    f, x, samples, weights:
        see arguments for :func:`fgivenx.compute_samples`
    """
    # convert to arrays
    if logZ is None:
        logZ = [0]
        f = [f]
        samples = [samples]
        weights = [weights]

    # logZ
    logZ = numpy.array(logZ, dtype='double')
    if len(logZ.shape) is not 1:
        raise ValueError("logZ should be a 1D array")

    # x
    x = numpy.array(x, dtype='double')
    if len(x.shape) is not 1:
        raise ValueError("x should be a 1D array")

    # f
    if len(logZ) != len(f):
            raise ValueError("len(logZ) = %i != len(f)= %i"
                             % (len(logZ), len(f)))
    for func in f:
        if not callable(func):
            raise ValueError("first argument f must be function"
                             "(or list of functions) of two variables")

    # samples
    if len(logZ) != len(samples):
            raise ValueError("len(logZ) = %i != len(samples)= %i"
                             % (len(logZ), len(samples)))
    samples = [numpy.array(s, dtype='double') for s in samples]
    for s in samples:
        if len(s.shape) is not 2:
            raise ValueError("each set of samples should be a 2D array")

    # weights
    if len(logZ) != len(weights):
            raise ValueError("len(logZ) = %i != len(weights)= %i"
                             % (len(logZ), len(weights)))
    weights = [numpy.array(w, dtype='double') if w is not None
               else numpy.ones(len(s), dtype='double')
               for w, s in zip(weights, samples)]

    for w, s in zip(weights, samples):
        if len(w.shape) is not 1:
            raise ValueError("each set of weights should be a 1D array")
        if len(w) != len(s):
            raise ValueError("len(w) = %i != len(s) = %i" % (len(s), len(w)))

    return logZ, f, x, samples, weights


def _normalise_weights(logZ, weights, ntrim=None):
    """ Correctly normalise the weights for trimming

    This takes a list of log-evidences, and re-normalises the weights so that
    the largest weight across all samples is 1, and the total weight in each
    set of samples is proportional to the evidence.

    Parameters
    ----------
    logZ: array-like
        log-evidences to weight each set of weights by

    weights: array-like of numpy.array
        list of not necessarily equal length list of weights

    Returns
    -------
    logZ: numpy.array
        evidences, renormalised so that max(logZ) = 0

    weights: list of 1D numpy.array
        normalised weights
    """
    logZ -= logZ.max()
    Zs = numpy.exp(logZ)
    weights = [w/w.sum()*Z for w, Z in zip(weights, Zs)]
    wmax = max([w.max() for w in weights])
    weights = [w/wmax for w in weights]
    ntot = sum([w.sum() for w in weights])
    if ntrim is not None and ntrim < ntot:
        weights = [w*ntrim/ntot for w in weights]
    return logZ, weights


def _equally_weight_samples(samples, weights):
    """ Convert samples to be equally weighted.

    Samples are trimmed by discarding samples in accordance with a probability
    determined by the corresponding weight.

    This function has assumed you have normalised the weights properly.
    If in doubt, convert weights via: `weights /= weights.max()`

    Parameters
    ----------
    samples: array-like
        Samples to trim.

    weights: array-like
        Weights to trim by.

    Returns
    -------
    1D numpy.array:
        Equally weighted sample array. `shape=(len(samples))`
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
