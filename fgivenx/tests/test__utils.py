import numpy
import pytest
import os
import fgivenx.io
import pytest
from shutil import rmtree
from numpy.testing import assert_allclose, assert_array_equal
import scipy.stats
import scipy.integrate
import scipy.special
from fgivenx._utils import _check_args, _normalise_weights, _equally_weight_samples

def test__check_args():
    numpy.random.seed(0)
    nfuncs = 3
    logZ = numpy.random.rand(nfuncs)
    f = [lambda x: x**i for i in range(nfuncs)]
    nx = 100
    x = numpy.linspace(0,1,nx)
    nsamps = 200
    nparams = 5
    samples = numpy.random.rand(nfuncs,nsamps,nparams)
    weights = numpy.random.rand(nfuncs,nsamps)

    # check these valid versions pass
    _check_args(logZ, f, x, samples, weights)
    _check_args(None, f[0], x, samples[0], weights[0])

    with pytest.raises(ValueError):
        _check_args(numpy.ones((2,2)), f, x, samples, weights)
    
    with pytest.raises(ValueError):
        _check_args(logZ, f, numpy.ones((2,2)), samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, numpy.ones((2,2)), samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f[1:], x, samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, numpy.ones_like(f), x, samples, weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples[1:], weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, numpy.random.rand(nfuncs,nparams), weights)

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples, weights[1:])

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples, numpy.random.rand(nfuncs))

    with pytest.raises(ValueError):
        _check_args(logZ, f, x, samples, numpy.random.rand(nfuncs,nsamps+1))
