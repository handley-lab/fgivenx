import numpy
import pytest
import os
import fgivenx.io
import pytest
from shutil import rmtree
from numpy.testing import assert_allclose
import scipy.stats
import scipy.integrate
from fgivenx.mass import PMF

def test_PMF():
    # Compute samples
    numpy.random.seed(0)
    nsamp = 100
    samples = numpy.concatenate((-5+numpy.random.randn(nsamp//2),5+numpy.random.randn(nsamp//2)))

    # Compute PMF
    y = numpy.random.uniform(-10,10,10)
    m = PMF(samples, y)

    # Compute PMF via monte carlo
    N = 100000
    kernel = scipy.stats.gaussian_kde(samples)
    s = kernel.resample(N)[0]
    m_ = [sum(kernel(s)<=kernel(y_i))/N for y_i in y]
    assert_allclose(m, m_, atol=3*N**-0.5)

    # Compute PMF via quadrature
    m_ = [scipy.integrate.quad(lambda x: kernel(x)*(kernel(x)<=kernel(y_i)), -numpy.inf, numpy.inf,limit=500)[0] for y_i in y]
    assert_allclose(m, m_, atol=1e-4)
    

