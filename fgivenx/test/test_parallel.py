import numpy
import pytest
from numpy.testing import assert_allclose
from fgivenx.parallel import parallel_apply


def f(x):
    return x**3


def test_parallel_apply():
    array = numpy.linspace(-1, 1, 100)

    with pytest.raises(TypeError):
        parallel_apply(f, array, wrong_argument=None)

    with pytest.raises(ValueError):
        parallel_apply(f, array, parallel='wrong_argument')

    for par in [False, True, -1, 2]:
        farray = parallel_apply(f, array, parallel=par)
        assert_allclose([f(x) for x in array], farray)
