import numpy
import pytest
import os
from shutil import rmtree
from numpy.testing import assert_allclose
import scipy.stats
import scipy.integrate
import scipy.special
from fgivenx.mass import PMF, compute_pmf


def gaussian_pmf(y, mu=0, sigma=1):
    return scipy.special.erfc(numpy.abs(y-mu)/numpy.sqrt(2)/sigma)


def test_gaussian():
    numpy.random.seed(0)
    nsamp = 5000
    samples = numpy.random.randn(nsamp)
    y = numpy.random.uniform(-3, 3, 10)
    m = PMF(samples, y)
    m_ = gaussian_pmf(y)
    assert_allclose(m, m_, rtol=3e-1)


def test_PMF():
    # Compute samples
    numpy.random.seed(0)
    nsamp = 100
    samples = numpy.concatenate((-5+numpy.random.randn(nsamp//2),
                                 5+numpy.random.randn(nsamp//2)))

    # Compute PMF
    y = numpy.random.uniform(-10, 10, 10)
    m = PMF(samples, y)

    # Compute PMF via monte carlo
    N = 100000
    kernel = scipy.stats.gaussian_kde(samples)
    s = kernel.resample(N)[0]
    m_ = [sum(kernel(s) <= kernel(y_i))/float(N) for y_i in y]
    assert_allclose(m, m_, atol=3*N**-0.5)

    # Compute PMF via quadrature
    m_ = [scipy.integrate.quad(lambda x: kernel(x)*(kernel(x) <= kernel(y_i)),
                               -numpy.inf, numpy.inf, limit=500)[0]
          for y_i in y]
    assert_allclose(m, m_, atol=1e-4)

    assert_allclose([0, 0], PMF(samples, [-1e3, 1e3]))

    samples = [0, 0]
    m = PMF(samples, y)
    assert_allclose(m, numpy.zeros_like(y))


def test_compute_pmf():

    with pytest.raises(TypeError):
        compute_pmf(None, None, wrong_argument=None)

    cache = '.test_cache/test'
    numpy.random.seed(0)
    nsamp = 5000
    a, b, e, f = 0, 1, 0, 1
    m = numpy.random.normal(a, b, nsamp)
    c = numpy.random.normal(e, f, nsamp)
    nx = 100
    x = numpy.linspace(-1, 1, nx)
    fsamps = (numpy.outer(x, m) + c)
    ny = 100
    y = numpy.linspace(-3, 3, ny)

    assert(not os.path.isfile(cache + '_masses.pkl'))
    m = compute_pmf(fsamps, y, cache=cache)
    assert(os.path.isfile(cache + '_masses.pkl'))

    m_ = [gaussian_pmf(y, a*xi+e, numpy.sqrt(b**2*xi**2+f**2)) for xi in x]
    assert_allclose(m.transpose(), m_, atol=3e-1)

    m = compute_pmf(fsamps, y, cache=cache)
    assert_allclose(m.transpose(), m_, atol=3e-1)

    rmtree('.test_cache')
