import numpy
import os
import pytest
from shutil import rmtree
from numpy.testing import assert_allclose, assert_almost_equal
from fgivenx.dkl import DKL, compute_dkl


def gaussian_dkl(mu, sig, mu_, sig_):
    return numpy.log(sig_/sig) + (sig**2+(mu-mu_)**2)/2/sig_**2 - 0.5


def test_DKL():
    numpy.random.seed(0)
    mu_, sig_ = 0, 1
    mu, sig = 0.2, 0.1
    size = 1000
    samples_ = numpy.random.normal(mu_, sig_, size)
    samples = numpy.random.normal(mu, sig, size)

    dkl = DKL((samples, samples_))
    dkl_true = gaussian_dkl(mu, sig, mu_, sig_)
    assert_almost_equal(dkl/dkl_true, 1, 1)


def test_compute_dkl():

    with pytest.raises(TypeError):
        compute_dkl(None, None, wrong_argument=None)

    cache = '.test_cache/test'
    numpy.random.seed(0)

    nx = 100
    x = numpy.linspace(-1, 1, nx)

    nsamp = 2000
    a, b, e, f = 0.1, 0.1, 0.1, 0.1
    m = numpy.random.normal(a, b, nsamp)
    c = numpy.random.normal(e, f, nsamp)
    fsamps = (numpy.outer(x, m) + c)

    a_, b_, e_, f_ = 0, 1, 0, 1
    m_ = numpy.random.normal(a_, b_, nsamp)
    c_ = numpy.random.normal(e_, f_, nsamp)
    prior_fsamps = (numpy.outer(x, m_) + c_)

    assert(not os.path.isfile(cache + '_dkl.pkl'))
    dkl = compute_dkl(fsamps, prior_fsamps, cache=cache)
    assert(os.path.isfile(cache + '_dkl.pkl'))

    dkl_ = [gaussian_dkl(a*xi+e,
                         numpy.sqrt(b**2*xi**2+f**2),
                         a_*xi+e_,
                         numpy.sqrt(b_**2*xi**2+f_**2)
                         ) for xi in x]
    assert_allclose(dkl, dkl_, atol=1e-1)

    dkl = compute_dkl(fsamps, prior_fsamps, cache=cache)
    assert_allclose(dkl, dkl_, atol=1e-1)

    rmtree('.test_cache')
