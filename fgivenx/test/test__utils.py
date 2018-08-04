import numpy
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from fgivenx._utils import _check_args, _normalise_weights, \
                           _equally_weight_samples


def test__check_args():
    numpy.random.seed(0)
    nfuncs = 3
    logZ = numpy.random.rand(nfuncs)
    f = [lambda x: x**i for i in range(nfuncs)]
    nx = 100
    x = numpy.linspace(0, 1, nx)
    nsamps = 200
    nparams = 5
    samples = numpy.random.rand(nfuncs, nsamps, nparams)
    weights = numpy.random.rand(nfuncs, nsamps)

    # check these valid versions pass
    _check_args(logZ, f, x, samples, weights)
    _check_args(None, f[0], x, samples[0], weights[0])

    with pytest.raises(ValueError):
        _check_args(numpy.ones((2, 2)), f, x, samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, numpy.ones((2, 2)), samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, numpy.ones((2, 2)), samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f[1:], x, samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, numpy.ones_like(f), x, samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples[1:], weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, numpy.random.rand(nfuncs, nparams), weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples, weights[1:])

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples, numpy.random.rand(nfuncs))

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples, numpy.random.rand(nfuncs, nsamps+1))


def assert_in_ratio(a, b):
    assert_almost_equal(numpy.array(a)*max(b), max(a)*numpy.array(b))


def test__normalise_weights():
    numpy.random.seed(0)
    nfuncs = 5
    nsamps = 5000
    ntrim = 500
    logZ = numpy.random.rand(nfuncs)
    weights = numpy.random.rand(nfuncs, nsamps)

    logZ, weights = _normalise_weights(logZ, weights)

    assert_almost_equal(numpy.max(weights), 1)
    assert_almost_equal(numpy.max(logZ), 0)
    assert_in_ratio([sum(w) for w in weights], numpy.exp(logZ))

    logZ, weights = _normalise_weights(logZ, weights, ntrim)

    assert_in_ratio([sum(w) for w in weights], numpy.exp(logZ))
    assert_almost_equal(numpy.sum(weights), ntrim)


def test__equally_weight_samples():
    numpy.random.seed(0)
    nsamps = 5000
    nparams = 5
    samples = numpy.random.rand(nsamps, nparams)
    weights = numpy.random.rand(nsamps)

    with pytest.raises(ValueError):
        _equally_weight_samples(samples, weights[1:])

    with pytest.raises(ValueError):
        _equally_weight_samples(samples, weights+1)

    with pytest.raises(ValueError):
        _equally_weight_samples(samples, weights-1)

    samples_1 = _equally_weight_samples(samples, weights)
    rand_1 = numpy.random.rand()
    samples_2 = _equally_weight_samples(samples, weights)
    rand_2 = numpy.random.rand()

    assert_allclose(samples_1, samples_2)
    assert(rand_1 != rand_2)
    assert_almost_equal(len(samples_1)/sum(weights), 1, 1)
