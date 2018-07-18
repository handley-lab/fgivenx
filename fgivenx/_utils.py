import numpy

def _check_args(logZ, f, x, samples, weights):
    """ Check the arguments for compute_samples. """
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


def _normalise_weights(logZ, weights, ntrim):
    """ Correctly normalise the weights for trimming"""
    Zs = numpy.exp(logZ-logZ.max())
    weights = [w/w.sum()*Z for w, Z in zip(weights, Zs)]
    wmax = max([w.max() for w in weights])
    weights = [w/wmax for w in weights]
    ntot = sum([w.sum() for w in weights])
    if ntrim is not None and ntrim < ntot:
        weights = [w*ntrim/ntot for w in weights]
    return logZ, weights
